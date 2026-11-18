# Embedding Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make five embedding models selectable per DVC experiment, embed each unique sentence once,
improve sentence splitting and training, and add a `score` stage that compares topics with Claude's
codes.

**Architecture:** Pure, testable helpers live in two new modules (`sentences.py` for splitting and
dedup, `embedding_models.py` for loading/encoding) and in small named functions inside the stage
scripts. Stage scripts (`prepare`, `embed`, `train`, `predict`, new `score`) stay thin `__main__`
orchestrators. `sentence_id` links `data/sentences.parquet` (unique) to `data/texts.parquet`
(occurrences); every stage joins on it instead of relying on row positions.

**Tech Stack:** Python 3.12, polars, NLTK, sentence-transformers 5.1, BERTopic 0.17.3, HDBSCAN,
UMAP, DVC (Hydra composition from `conf/`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-embedding-model-comparison-design.md`

## Global Constraints

- Readability is a hard requirement: plain named steps, short functions with docstrings, no clever
  index juggling. Adam must be able to follow the unique-sentence logic.
- Do **not** run prepare/embed/train on all local posts. Only unit tests and the smoke run in Task 9
  (`prepare.settings.nobs=300`, `model2vecdk`).
- Always L2-normalize embeddings after truncation, in our own code.
- Never float16 for `google/embeddinggemma-300m`.
- Same code path locally and on STAR; STAR is offline and loads models from `embed.settings.model_dir`.
- Output files read by `evaluate.py` keep their names and columns (`ann_id` = `post_id`).
- Hydra is enabled (`.dvc/config`): `conf/` is the source of truth; `dvc exp run` regenerates
  `params.yaml`. Every conf change must also be reflected in `params.yaml` so plain `dvc repro`
  still works.
- Ruff line length 120; keep `make check` clean.
- Run tests with `uv run python -m pytest`. Use `PYTHONIOENCODING=utf-8` on Windows when printing
  Danish text.

## Review Focus

1. **Post with no Title / empty Text** (null in Excel): must not crash; the post contributes only
   what it has. → test in Task 2 (`test_title_rows_skip_empty_titles`).
2. **A sentence that appears in many posts and in both train and non-train sets**: training flag is
   per unique sentence; every occurrence gets the same topic. → test in Task 5
   (`test_join_on_sentences_gives_every_copy_the_same_topic`).
3. **Claude quote that spans two sentences or comes from the title**: must still match. → tests in
   Task 8 (`test_match_quote_spanning_sentences`, title row present in fixture).
4. **Claude item with no positive posts** (e.g. `AI` with 2/100 in the pilot, possibly 0): F1 must be
   skipped, not counted as 0 or crash the mean. → test in Task 8 (`test_best_topic_f1_no_positives`).
5. **UMAP with a fixed `random_state` runs single-threaded**: fine locally, very slow at 8M sentences
   on STAR. A `train.settings.umap_seed` switch turns it off, and the UMAP config comment says to use
   it on STAR. → Task 6 Step 3.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/job_post_topic_modelling/sentences.py` (new) | HTML cleanup, splitting, heading rule, sensitive filter, title rows, dedup |
| `src/job_post_topic_modelling/prepare.py` | Load Excel/STAR, language filter, call `sentences.py`, write `texts.parquet` + `sentences.parquet` |
| `src/job_post_topic_modelling/embedding_models.py` (new) | dtype rules, model path, `load_embedding_model`, `truncate_and_normalize`, `encode` |
| `src/job_post_topic_modelling/embed.py` | Embed `sentences.parquet` in shards, write cost metrics |
| `src/job_post_topic_modelling/utils/data_io.py` | Shard loading (now with `rows=`), `join_on_sentences` |
| `src/job_post_topic_modelling/train.py` | Relative cluster sizes, random sample, seeds, outlier option, outputs |
| `src/job_post_topic_modelling/predict.py` | Predict per unique sentence, join back |
| `src/job_post_topic_modelling/score.py` (new) | Claude-code metrics and `topics_overview.csv` |
| `conf/embed/model/*.yaml`, `conf/embed/settings/*.yaml`, `conf/train/**`, `conf/score/default.yaml` (new), `conf/config.yaml`, `params.yaml`, `dvc.yaml` | Config and stages |
| `tests/test_sentences.py`, `tests/test_embedding_models.py`, `tests/test_train.py`, `tests/test_predict.py`, `tests/test_score.py` (new) | Unit tests |

---

### Task 1: Test setup and sentence splitting

**Files:**
- Delete: `tests/test_foo.py` (imports a module that does not exist, so the suite fails today)
- Create: `src/job_post_topic_modelling/sentences.py`
- Test: `tests/test_sentences.py`

**Interfaces:**
- Produces:
  - `html_to_text(text: str) -> str`
  - `split_into_sentences(text: str, unit: str = "sentence") -> list[str]` (`unit` is `"sentence"` or `"paragraph"`)
  - `is_heading(sentence: str) -> bool`
  - `is_sensitive(sentence: str) -> bool`

- [ ] **Step 1: Install the environment and remove the broken test**

Run: `uv sync` then `git rm tests/test_foo.py`
Expected: `uv run python -m pytest` reports "no tests ran" (exit code 5), not an import error.

- [ ] **Step 2: Write failing tests in `tests/test_sentences.py`**

```python
from job_post_topic_modelling.sentences import html_to_text, is_heading, is_sensitive, split_into_sentences

def test_html_list_tags_become_line_breaks():
    assert html_to_text("<ul><li>Løn</li><li>Pension</li></ul>").split("\n") == ["Løn", "Pension"]

def test_html_removes_x0009_and_extra_spaces_but_keeps_line_breaks():
    assert html_to_text("Vi  har_x0009_flekstid.\nGod løn.") == "Vi har flekstid.\nGod løn."

def test_single_line_break_list_is_split():
    text = "Du er struktureret\nDu taler engelsk\nDu har kørekort"
    assert split_into_sentences(text) == ["Du er struktureret", "Du taler engelsk", "Du har kørekort"]

def test_bullets_and_sentences_are_split():
    assert split_into_sentences("Vi tilbyder flekstid. God pension.•Frokostordning") == [
        "Vi tilbyder flekstid.", "God pension.", "Frokostordning"]

def test_paragraph_unit_keeps_lines_within_a_blank_line_block_together():
    assert split_into_sentences("Linje et.\nLinje to.\n\nNy blok.", unit="paragraph") == [
        "Linje et. Linje to.", "Ny blok."]

def test_headings_are_detected():
    for heading in ["Vi tilbyder:", "Hvem er du?", "Om opgaven", "Din profil", "Dine arbejdsopgaver inkluderer:"]:
        assert is_heading(heading)

def test_short_sentences_with_content_are_kept():
    for sentence in ["Vi har flekstid.", "fastansættelse.", "Grib chancen nu!", "Løn efter kvalifikationer.",
                     "Frokostordning", "Fleksible arbejdstider", "Du taler engelsk",
                     "Har du lyst til at arbejde med børn og unge?"]:
        assert not is_heading(sentence)

def test_phone_formats_are_sensitive():
    assert is_sensitive("Ring på 3634 4990")
    assert is_sensitive("Ring på +45 36 34 49 90")
    assert is_sensitive("Ring på 36344990")
    assert not is_sensitive("Vi har 2024 ansatte og 1200 kunder")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_sentences.py -v`
Expected: FAIL with `ModuleNotFoundError: job_post_topic_modelling.sentences`

- [ ] **Step 4: Implement the four functions in `sentences.py`**

- `html_to_text`: replace `<br>`, `<li>`, `</li>`, `</p>`, `</div>` (case-insensitive, with attributes) with `\n`, then other tags with a space, `_x0009_` with a space, `[ \t\xa0]+` with one space, strip spaces around each line, and drop empty lines.
- `split_into_sentences`: `"sentence"` splits on `\r?\n|•|\*`, then `nltk.tokenize.sent_tokenize(part, language="danish")`. `"paragraph"` splits on the old rule `(?:\r?\n){2,}|•|\*`, sentence-tokenizes each block and joins the sentences of one block with a space. Both collapse inner whitespace and drop empty strings.
- `is_heading`: true if any of these holds:
  1. It ends with `:`.
  2. It has ≤4 words and ends with `?`.
  3. Lowercased and stripped of trailing `:?.!`, it is in a module-level `KNOWN_HEADINGS` set. Start the set with: om opgaven, om jobbet, om stillingen, stillingen, om dig, om os, om virksomheden, din profil, dine opgaver, arbejdsopgaver, kvalifikationer, faglige kvalifikationer, personlige kvalifikationer, vi tilbyder, vi tilbyder dig, vi forventer, vi søger, hvem er du, løn og ansættelsesvilkår, ansættelsesvilkår, ansøgning, yderligere oplysninger, kontakt, praktiske oplysninger.

  A bare "short with no full stop" rule is deliberately avoided, because it would drop list items like "Frokostordning".
- `is_sensitive`: move the date, e-mail and homepage patterns and `phone_pattern` from `prepare.py` unchanged, and add `\b\d{4}\s\d{4}\b` and `\+45[\s\d]{8,}`.
- Move the NLTK data-path lines from `prepare.py` (the `whoami` + `nltk.data.path.append`) here, since this module now calls NLTK.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_sentences.py -v`
Expected: 8 passed. If `sent_tokenize` needs data, run `uv run python -c "import nltk; nltk.download('punkt_tab')"` once.

- [ ] **Step 6: Commit**

```bash
git add src/job_post_topic_modelling/sentences.py tests/test_sentences.py
git commit -m "feat: sentence splitting with line-break lists, heading rule and phone filter"
```

---

### Task 2: Post splitting, title rows and unique sentences

**Files:**
- Modify: `src/job_post_topic_modelling/sentences.py`
- Test: `tests/test_sentences.py`

**Interfaces:**
- Consumes: Task 1 functions.
- Produces:
  - `split_posts(posts: pl.DataFrame, unit: str = "sentence") -> pl.DataFrame`. Input columns `post_id`, `text`; output columns `post_id`, `label`, `text`, `drop_reason` (null, `"heading"` or `"sensitive"`). `label` is `f"{post_id}_{i:02d}"` over the post's split parts in order (on STAR the input `post_id` column holds the section id, see Task 3).
  - `title_rows(posts: pl.DataFrame) -> pl.DataFrame`. Input `post_id`, `title`; one row per post with a non-empty title: `post_id`, `label=f"{post_id}_title"`, `text`, `drop_reason=null`. Titles are never checked with `is_heading`.
  - `deduplicate(texts: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]`. Returns `(texts_with_sentence_id, sentences)`. `sentences` has `sentence_id` (u32, 0..n-1 in first-seen order), `text`, `n_occurrences`. `texts_with_sentence_id` is the input plus `sentence_id`, same row order.

- [ ] **Step 1: Write failing tests**

```python
import polars as pl
from job_post_topic_modelling.sentences import deduplicate, split_posts, title_rows

def test_split_posts_marks_drop_reasons_and_labels():
    posts = pl.DataFrame({"post_id": ["p1"], "text": ["Vi tilbyder:\nFlekstid.\nRing på 3634 4990."]})
    out = split_posts(posts)
    assert out["label"].to_list() == ["p1_00", "p1_01", "p1_02"]
    assert out["drop_reason"].to_list() == ["heading", None, "sensitive"]

def test_title_rows_skip_empty_titles():
    posts = pl.DataFrame({"post_id": ["p1", "p2", "p3"], "title": ["Pædagog", None, " "]})
    out = title_rows(posts)
    assert out.select("post_id", "label", "text").rows() == [("p1", "p1_title", "Pædagog")]

def test_deduplicate_round_trip():
    texts = pl.DataFrame({"post_id": ["a", "a", "b"], "label": ["a_00", "a_01", "b_00"],
                          "text": ["Vi tilbyder flekstid.", "God løn.", "Vi tilbyder flekstid."]})
    texts_out, sentences = deduplicate(texts)
    assert sentences.rows() == [(0, "Vi tilbyder flekstid.", 2), (1, "God løn.", 1)]
    assert texts_out["sentence_id"].to_list() == [0, 1, 0]
    rebuilt = texts_out.join(sentences, on="sentence_id", suffix="_unique")
    assert (rebuilt["text"] == rebuilt["text_unique"]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_sentences.py -v`
Expected: 3 new FAIL with `ImportError`.

- [ ] **Step 3: Implement `split_posts`, `title_rows`, `deduplicate`**

`split_posts` loops over posts in plain Python (as today), applies `html_to_text` then `split_into_sentences`, and sets `drop_reason` with `is_heading` checked first, then `is_sensitive`. Null text is treated as an empty string. `deduplicate` uses `texts["text"].unique(maintain_order=True)` to build `sentences`, then a join on `text` that keeps the input order (`maintain_order="left"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_sentences.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/job_post_topic_modelling/sentences.py tests/test_sentences.py
git commit -m "feat: split posts with drop reasons, title rows and sentence dedup"
```

---

### Task 3: Rewire `prepare.py`

**Files:**
- Modify: `src/job_post_topic_modelling/prepare.py`
- Modify: `conf/prepare/star/star.yaml`, `conf/prepare/star/local.yaml`, `params.yaml`, `dvc.yaml` (prepare stage), `data/.gitignore`
- Test: `tests/test_sentences.py`

**Interfaces:**
- Consumes: Task 2 functions.
- Produces: `data/texts.parquet` (`post_id`, `label`, `text`, `sentence_id`; only kept rows) and `data/sentences.parquet` (`sentence_id`, `text`, `n_occurrences`). Also `post_id_from_section_id(section_ids: pl.Expr) -> pl.Expr` in `sentences.py`.

- [ ] **Step 1: Write a failing test for STAR post ids**

```python
from job_post_topic_modelling.sentences import post_id_from_section_id

def test_post_id_from_section_id():
    df = pl.DataFrame({"section_id": ["12345_s1", "12345_s2", "987_s10"]})
    assert df.select(post_id_from_section_id(pl.col("section_id")))["section_id"].to_list() == ["12345", "12345", "987"]
```

- [ ] **Step 2: Run it to verify it fails**, then implement it with regex `^(\d+)_s` and run it again. Expected: PASS.

- [ ] **Step 3: Rewrite the loading part of `prepare.py`**

- `load_excel(file_path: Path, par: DictConfig) -> pl.DataFrame` reads `ID`, `Title`, `Text` and returns `post_id`, `title`, `text`. Fixes the global `par`.
- `load_star_data(par)` returns `post_id`, `title`, `text`, plus `section_id` when sectioned data is used. `post_id` is `ann_id` if the parquet schema has it (check with `pl.scan_parquet(...).collect_schema()`), otherwise `post_id_from_section_id(section_id)`. `title` comes from `par.star.title_col` (new key, `null` by default in `star.yaml`). When that is null, title is all null and the script prints `"    - No title column on STAR, titles skipped"`.
- `clean_data`: keep `rename_jobcenter_obs` (on `post_id`) and `detect_language`. Remove the HTML regexes, which `split_posts` now does.

- [ ] **Step 4: Rewrite the splitting part of `__main__` as named steps**

1. `unit = "sentence" if par.settings.split_sentences else "paragraph"`. Splitting always runs.
2. `parts = split_posts(posts with label base, unit)`. On STAR, pass `section_id` as the `post_id` column so labels stay unique, then put the real `post_id` back by joining on the label prefix. Simpler alternative: add an optional `label_base` column to `split_posts`, used instead of `post_id` when present, and document it in its docstring. Pick that one.
3. Print counts per `drop_reason`, plus 20 random dropped headings (`seed=1`).
4. `texts = pl.concat([title_rows(posts), parts.filter(drop_reason is null)]).drop("drop_reason")`.
5. `texts, sentences = deduplicate(texts)`. Print the unique count and the duplicate share.
6. Write `data/texts.parquet` and `data/sentences.parquet`.

Delete `filter_sentences_remove_sensitive` and the `if False:` debug block.

- [ ] **Step 5: Update config and stage**

- `conf/prepare/star/star.yaml` gets `usestar: 1` and `title_col: null`. `local.yaml` gets `usestar: 0` and `title_col: null`. Mirror both in `params.yaml`.
- `dvc.yaml` prepare: add `src/job_post_topic_modelling/sentences.py` to deps and `data/sentences.parquet` to outs.
- `data/.gitignore`: add `/sentences.parquet`.

- [ ] **Step 6: Run the tests and a lint check**

Run: `uv run python -m pytest -v` and `uv run ruff check src tests`
Expected: all pass, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/job_post_topic_modelling/prepare.py src/job_post_topic_modelling/sentences.py tests conf/prepare params.yaml dvc.yaml data/.gitignore
git commit -m "feat: prepare writes texts and unique sentences with post_id and titles"
```

---

### Task 4: Embedding model configs and loader

**Files:**
- Create: `src/job_post_topic_modelling/embedding_models.py`
- Create: `conf/embed/model/f2llm_330m.yaml`, `f2llm_160m.yaml`, `embeddinggemma.yaml`, `e5_large_instruct.yaml`, `model2vecdk.yaml`
- Modify: `conf/embed/model/sentence_transformers_big.yaml`, `sentence_transformers_small.yaml`, `intfloat.yaml`, `GritLM.yaml`, `conf/embed/settings/*.yaml`, `params.yaml`
- Test: `tests/test_embedding_models.py`

**Interfaces:**
- Produces:
  - `resolve_device(device: str | None) -> str` returns `"cuda"` if available when `device` is None, else `"cpu"`.
  - `resolve_dtype(dtype: str | None, device: str, model_name: str) -> torch.dtype | None`. `None` means "library default, pass no `model_kwargs`".
  - `resolve_model_path(model_name: str, model_dir: str | None) -> str`
  - `load_embedding_model(model_par: DictConfig, settings: DictConfig, device: str | None = None) -> SentenceTransformer`. The `device` argument overrides `settings.device` (train/predict pass `"cpu"`).
  - `truncate_and_normalize(embeddings: np.ndarray, truncate_dim: int | None) -> np.ndarray`
  - `encode(model: SentenceTransformer, sentences: list[str], model_par: DictConfig, **encode_kwargs) -> np.ndarray` returns float32 vectors, truncated and normalized.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np, pytest, torch
from job_post_topic_modelling.embedding_models import resolve_dtype, resolve_model_path, truncate_and_normalize

def test_truncate_then_normalize():
    emb = np.array([[3.0, 4.0, 12.0], [0.0, 0.0, 5.0]])
    out = truncate_and_normalize(emb, truncate_dim=2)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    np.testing.assert_allclose(out[1], [0.0, 0.0])  # zero vector stays zero, no NaN

def test_full_size_is_normalized():
    out = truncate_and_normalize(np.array([[3.0, 4.0]]), truncate_dim=None)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0])

def test_auto_dtype():
    assert resolve_dtype("auto", "cuda", "x") == torch.bfloat16
    assert resolve_dtype("auto", "cpu", "x") == torch.float32
    assert resolve_dtype(None, "cpu", "x") is None

def test_gemma_refuses_float16():
    with pytest.raises(ValueError, match="float16"):
        resolve_dtype("float16", "cuda", "google/embeddinggemma-300m")

def test_model_path():
    assert resolve_model_path("google/embeddinggemma-300m", None) == "google/embeddinggemma-300m"
    assert resolve_model_path("google/embeddinggemma-300m", "/models").replace("\\", "/") == "/models/google/embeddinggemma-300m"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_embedding_models.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `embedding_models.py`**

`load_embedding_model`:
- If `model_dir` is set, it sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- It builds `SentenceTransformer(path, device=..., truncate_dim=model_par.truncate_dim, model_kwargs={"torch_dtype": dtype} if dtype else None)`.
- When `model_par.prompt` is set, it also passes `prompts={"clustering": prompt}` and `default_prompt_name="clustering"`. This way every `encode` call, including BERTopic's own calls on seed words, uses the same prompt.
- It sets `model.max_seq_length` when `model_par.max_seq_length` is set.
- It keeps the existing `use_model2vec` branch, which distils with `StaticEmbedding.from_distillation(path, device, pca_dims)`.
- `encode` calls `model.encode(..., convert_to_numpy=True)`, then `truncate_and_normalize`, then `.astype(np.float32)`.

- [ ] **Step 4: Write the model configs**

All configs have the fields `embedding_model`, `prompt`, `truncate_dim`, `max_seq_length`, `dtype`, `use_model2vec`, `pca_dims`. Values (copy exactly):

| File | embedding_model | prompt | truncate_dim | max_seq_length | dtype |
|---|---|---|---|---|---|
| `sentence_transformers_big` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | null | null | 128 | auto |
| `sentence_transformers_small` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | null | null | 128 | auto |
| `f2llm_330m` | `codefuse-ai/F2LLM-v2-330M` | null | 256 | 256 | bfloat16 |
| `f2llm_160m` | `codefuse-ai/F2LLM-v2-160M` | null | 256 | 256 | bfloat16 |
| `embeddinggemma` | `google/embeddinggemma-300m` | `"task: clustering \| query: "` | null | 256 | auto |
| `e5_large_instruct` | `intfloat/multilingual-e5-large-instruct` | `"Instruct: Identify the topic of this sentence from a Danish job post\nQuery: "` | null | 256 | auto |
| `model2vecdk` | `andersborges/model2vecdk` | null | null | null | null |

- `intfloat.yaml` and `GritLM.yaml` keep `use_model2vec: True` and their `pca_dims`, with all new fields set to `null`.
- `use_model2vec` is false everywhere else. `pca_dims` keeps its current value where it already exists, and is the full size otherwise.
- In YAML the `|` in the Gemma prompt is literal. Use double quotes so `\n` in the e5 prompt is a real newline.
- Add `model_dir: null` to `conf/embed/settings/default.yaml`, `optimize.yaml` and `optimize_sentence.yaml`, with a comment: on STAR, set to the folder that holds the copied models, e.g. `/home/<user>@PROD.SITAD.DK/code/help/installations/`.
- Mirror the default (`sentence_transformers_big`) and the settings in `params.yaml`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_embedding_models.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/job_post_topic_modelling/embedding_models.py tests/test_embedding_models.py conf/embed params.yaml
git commit -m "feat: configurable embedding models with prompts, Matryoshka size and dtype"
```

---

### Task 5: `embed.py` on unique sentences, shard loading by rows, joining back

**Files:**
- Modify: `src/job_post_topic_modelling/embed.py`, `src/job_post_topic_modelling/utils/data_io.py`, `dvc.yaml` (embed stage)
- Test: `tests/test_embedding_models.py`, `tests/test_train.py` (new)

**Interfaces:**
- Consumes: Task 4 `load_embedding_model`, `encode`. Task 3 output files.
- Produces:
  - `embedding_cost(n_sentences: int, seconds: float, vector_dim: int, n_projected: int = 56_000_000) -> dict[str, float]` in `embed.py`, with keys `sentences_per_second`, `vector_dim`, `bytes_per_vector`, `projected_storage_gb`.
  - `load_pretrained_embeddings(filepath, nobs=None, rows: np.ndarray | None = None) -> np.ndarray` in `data_io.py`. When `rows` is given (sorted global row numbers), each shard keeps only its rows, and the result is in `rows` order.
  - `join_on_sentences(texts: pl.DataFrame, per_sentence: pl.DataFrame) -> pl.DataFrame` in `data_io.py`. It does an inner join on `sentence_id`, keeps the `texts` row order, and adds `ann_id = post_id`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embedding_models.py
from job_post_topic_modelling.embed import embedding_cost

def test_embedding_cost():
    cost = embedding_cost(n_sentences=1000, seconds=2.0, vector_dim=256)
    assert cost == {"sentences_per_second": 500.0, "vector_dim": 256, "bytes_per_vector": 1024,
                    "projected_storage_gb": 57.344}
```

```python
# tests/test_train.py
import numpy as np, polars as pl
from job_post_topic_modelling.utils.data_io import join_on_sentences, load_pretrained_embeddings

def test_load_selected_rows_across_shards(tmp_path):
    np.save(tmp_path / "embeddings_shard_1.npy", np.arange(6).reshape(3, 2))
    np.save(tmp_path / "embeddings_shard_2.npy", np.arange(6, 10).reshape(2, 2))
    out = load_pretrained_embeddings(tmp_path, rows=np.array([1, 3]))
    np.testing.assert_array_equal(out, [[2, 3], [6, 7]])

def test_join_on_sentences_gives_every_copy_the_same_topic():
    texts = pl.DataFrame({"post_id": ["a", "b", "a"], "label": ["a_00", "b_00", "a_01"], "sentence_id": [0, 0, 1]})
    per_sentence = pl.DataFrame({"sentence_id": [1, 0], "predicted_topic": [7, 3]})
    out = join_on_sentences(texts, per_sentence)
    assert out["label"].to_list() == ["a_00", "b_00", "a_01"]
    assert out["predicted_topic"].to_list() == [3, 3, 7]
    assert out["ann_id"].to_list() == ["a", "b", "a"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_embedding_models.py tests/test_train.py -v`
Expected: FAIL (`ImportError` / unexpected keyword `rows`).

- [ ] **Step 3: Implement**

- `embedding_cost` assumes float32 (4 bytes per number) and uses `projected_storage_gb = n_projected * bytes_per_vector / 1e9`, rounded to 3 decimals.
- `embed.py` `__main__` reads `data/sentences.parquet["text"]`, calls `load_embedding_model(par.model, par.settings)`, times `embed_in_shards`, and writes the `embed.json` metrics: the existing time entry plus `embedding_cost(...)`.
- `embed_in_shards` calls `encode(...)` instead of `model.encode`.
- Delete `get_embedding_model_name` and `get_embedding_model`.
- `dvc.yaml` embed: the deps replace `data/texts.parquet` with `data/sentences.parquet` and add `embedding_models.py`. Make `output/metrics/embed.json` a `metrics:` out with `cache: false`, so `dvc exp show` lists it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/job_post_topic_modelling/embed.py src/job_post_topic_modelling/utils/data_io.py tests dvc.yaml
git commit -m "feat: embed unique sentences and record embedding cost"
```

---

### Task 6: `train.py` improvements

**Files:**
- Modify: `src/job_post_topic_modelling/train.py`, `conf/train/clustering/HDBSCAN.yaml`, `conf/train/dimensionality_reduction/UMAP.yaml`, `conf/train/settings/default.yaml`, `params.yaml`, `dvc.yaml` (train stage)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `load_embedding_model` (Task 4), `load_pretrained_embeddings(rows=)` and `join_on_sentences` (Task 5).
- Produces:
  - `resolve_cluster_sizes(n_train: int, min_cluster_size_share: float, min_samples_ratio: float, floor: int = 10) -> tuple[int, int]`
  - `sample_training_ids(n_sentences: int, nobs: int | None, seed: int) -> np.ndarray` returns sorted unique ids. If `nobs` is None or ≥ `n_sentences`, it returns all ids.
  - `get_dimensionality_reduction_model(par, seed: int, n_jobs: int)` and `get_clustering_model(par, n_train: int, seed: int, n_jobs: int)`
  - File `models/submodels/training_sentence_ids.npy`
  - `output/predicted_topics_train.parquet`: per occurrence of a training sentence, with columns `post_id`, `label`, `text`, `sentence_id`, `predicted_topic`, `topic_probability`, `ann_id`, `training_data=True`.

- [ ] **Step 1: Write failing tests**

```python
from job_post_topic_modelling.train import resolve_cluster_sizes, sample_training_ids

def test_resolve_cluster_sizes_scales_with_data():
    assert resolve_cluster_sizes(8_000_000, 0.0005, 0.1) == (4000, 400)
    assert resolve_cluster_sizes(180_000, 0.0005, 0.1) == (90, 9)
    assert resolve_cluster_sizes(1_000, 0.0005, 0.1) == (10, 1)  # floor

def test_sample_training_ids():
    ids = sample_training_ids(100, nobs=10, seed=42)
    assert len(ids) == 10 and len(set(ids)) == 10 and list(ids) == sorted(ids)
    np.testing.assert_array_equal(ids, sample_training_ids(100, nobs=10, seed=42))
    np.testing.assert_array_equal(sample_training_ids(5, nobs=None, seed=42), np.arange(5))
    np.testing.assert_array_equal(sample_training_ids(5, nobs=50, seed=42), np.arange(5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_train.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Update configs**

- `HDBSCAN.yaml`: replace `min_cluster_size`/`min_samples` with `min_cluster_size_share: 0.0005` and `min_samples_ratio: 0.1`, and drop `core_dist_n_jobs`.
- `UMAP.yaml`: drop `n_jobs` and `random_state`, and add the comment: "random_state comes from train.settings.seed; a fixed seed makes UMAP single-threaded. On STAR pass `-S train.settings.umap_seed=false` for speed."
- `settings/default.yaml`: add `seed: 42`, `umap_seed: true`, `n_jobs: 8`, `reduce_outliers: null` (comment: null = off, or a BERTopic strategy such as `embeddings`), `reduce_outliers_threshold: 0.0`. Add the comment "nobs counts unique sentences".
- Mirror all of this in `params.yaml`.


- [ ] **Step 4: Implement in `train.py`**

- `resolve_cluster_sizes` uses `max(floor, round(share * n))` and `max(1, round(ratio * min_cluster_size))`.
- `sample_training_ids` uses `np.random.default_rng(seed).choice(n, nobs, replace=False)`, sorted.
- `get_clustering_model` passes the resolved sizes and `core_dist_n_jobs=n_jobs` to HDBSCAN, and `random_state=seed` to KMeans.
- UMAP gets `random_state=seed if par.settings.umap_seed else None`, plus `n_jobs`.
- Keep everything in `__main__` as named steps:
  1. Load `sentences.parquet` and `texts.parquet`.
  2. `ids = sample_training_ids(sentences.height, par.settings.nobs, par.settings.seed)`.
  3. Load embeddings for those rows only, and the documents for those sentences.
  4. Print the resolved cluster sizes.
  5. Fit BERTopic.
  6. If `reduce_outliers` is set, call `topic_model.reduce_outliers(docs, topics, strategy=..., embeddings=..., threshold=...)`, then `topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer_model)`.
  7. Save the model, the submodels (with `with open(...)`) and `training_sentence_ids.npy`.
  8. Build `per_sentence` (`sentence_id`, `predicted_topic`, `topic_probability`) and `join_on_sentences(texts, per_sentence)`. The join is inner, so only training sentences remain. Add `training_data=True`.
  9. Write the two parquet outputs, using the existing `aggregate_predictions_to_ann_level`.
  10. Write `train.json` with the time, `n_train_sentences`, `min_cluster_size`, `min_samples`.
- The embedding model is `load_embedding_model(full_par.embed.model, full_par.embed.settings, device="cpu")`.
- Delete `rescale`, `UnsupportedAnchorTypeError`, `get_embedding_model_cpu`, and the unused parameters `embeddings` and `stop_words`.
- `dvc.yaml` train:
  - deps: add `data/sentences.parquet`, `embedding_models.py`.
  - outs: add `models/submodels/training_sentence_ids.npy`.
  - Make `output/metrics/train.json` a `metrics:` out with `cache: false`.
  - The `*_NUM_THREADS` values stay 8, matching `n_jobs: 8`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest -v` and `uv run ruff check src tests`
Expected: all pass, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/job_post_topic_modelling/train.py conf/train params.yaml dvc.yaml tests/test_train.py
git commit -m "feat: relative cluster sizes, random training sample, seeds and outlier option"
```

---

### Task 7: `predict.py` per unique sentence

**Files:**
- Modify: `src/job_post_topic_modelling/predict.py`, `dvc.yaml` (predict stage)
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `join_on_sentences`, `load_pretrained_embeddings` (Task 5), `load_embedding_model` (Task 4), `training_sentence_ids.npy` (Task 6).
- Produces (same files as today):
  - `output/predicted_topics.parquet`, now with an `ann_id` column.
  - `output/predicted_topics_agg.parquet`
  - New helpers:
    - `topic_similarities(embeddings: np.ndarray, topic_embeddings: np.ndarray, ignore_limit: float | None) -> np.ndarray`: cosine similarities with negatives set to 0, and rows whose max is below `ignore_limit` set to all 0.
    - `best_topics(similarities: np.ndarray, ignore_limit: float | None) -> tuple[np.ndarray, np.ndarray]`: the argmax topic and its similarity. Rows whose max is below `ignore_limit` get topic -1 and probability 1.0, as today.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from job_post_topic_modelling.predict import best_topics, topic_similarities

def test_topic_similarities_clip_and_ignore():
    emb = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]])
    topics = np.array([[1.0, 0.0], [0.0, 1.0]])
    sims = topic_similarities(emb, topics, ignore_limit=0.5)
    np.testing.assert_allclose(sims, [[1.0, 0.0], [0.6, 0.8], [0.0, 0.0]])

def test_best_topics_marks_low_similarity_as_outlier():
    sims = np.array([[0.9, 0.1], [0.2, 0.3]])
    topics, probs = best_topics(sims, ignore_limit=0.5)
    assert topics.tolist() == [0, -1]
    np.testing.assert_allclose(probs, [0.9, 1.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_predict.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

- Replace the inline similarity code in both branches with the two helpers.
- `__main__` works on `sentences.parquet` (the first `predict.settings.nobs` unique sentences when it is set) and their embeddings. It builds `per_sentence` with `sentence_id`, the topic or `topic_i` columns, and `training_data = sentence_id.is_in(training_ids)`, then calls `join_on_sentences(texts, per_sentence)`.
- Keep batching over unique sentences with the existing `batch_mode`/`shard_size` logic.
- `add_wfh` uses the `ann_id` column directly. Delete `extract_ann_id`.
- Read `use_cuml` with `par_train.dimensionality_reduction.get("use_cuml", False)`.
- Embedding model: `load_embedding_model(..., device="cpu")`.
- `dvc.yaml` predict deps: add `data/texts.parquet`, `data/sentences.parquet`, `data/embeddings/`, `models/submodels/training_sentence_ids.npy`, `embedding_models.py`. Today these are missing even though they are read.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/job_post_topic_modelling/predict.py tests/test_predict.py dvc.yaml
git commit -m "feat: predict per unique sentence and join back to occurrences"
```

---

### Task 8: `score.py` and the `score` stage

**Files:**
- Create: `src/job_post_topic_modelling/score.py`, `conf/score/default.yaml`
- Modify: `conf/config.yaml`, `params.yaml`, `dvc.yaml`, `data/.gitignore`
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: `output/predicted_topics_train.parquet` (Task 6), `models/bertopic_model` (Task 6), Claude `results.parquet` (`ID`, `valid`, each item, `<item>_evidence`, `<item>_quote_ok`).
- Produces:
  - `load_claude_codes(path: Path) -> pl.DataFrame`: `valid == 1` only, `ID` renamed to `post_id`.
  - `item_names(codes: pl.DataFrame) -> list[str]`: columns ending in `_quote_ok`, with that suffix removed. The same rule as `scripts/llm_topics/annotate_jobnet.py`.
  - `post_topics(occurrences: pl.DataFrame) -> pl.DataFrame`: distinct (`post_id`, `predicted_topic`), topic -1 excluded.
  - `best_topic_f1(post_topics: pl.DataFrame, codes: pl.DataFrame, item: str) -> tuple[int | None, float | None]`
  - `match_quote(quote: str, sentences: list[str]) -> list[int]`: positions in `sentences` that contain the quote, or that are contained in it (after lowercasing and collapsing whitespace).
  - `evidence_shares(occurrences: pl.DataFrame, codes: pl.DataFrame, items: list[str]) -> dict[str, float]`: keys `evidence_in_outliers_share` and `evidence_unmatched_share`.
  - `topic_diversity(top_words: list[list[str]], n: int = 10) -> float`
  - Outputs `output/metrics/score.json` and `output/topics_overview.csv`.

- [ ] **Step 1: Write failing tests**

```python
import math, polars as pl
from job_post_topic_modelling.score import best_topic_f1, evidence_shares, match_quote, post_topics, topic_diversity

CODES = pl.DataFrame({"post_id": ["A", "B", "C", "D"], "flex": [1, 1, 0, 0], "flex_evidence": ["flekstid", "fleksibel arbejdstid", "", ""],
                      "AI": [0, 0, 0, 0], "AI_evidence": ["", "", "", ""]})
OCC = pl.DataFrame({
    "post_id": ["A", "A", "B", "B", "C", "D"],
    "text": ["Pædagog", "Vi har flekstid.", "Du får fleksibel", "arbejdstid og frihed.", "God løn.", "Kantine."],
    "label": ["A_title", "A_00", "B_00", "B_01", "C_00", "D_00"],
    "predicted_topic": [2, 1, -1, 1, 1, 3]})

def test_best_topic_f1():
    topic, f1 = best_topic_f1(post_topics(OCC), CODES, "flex")
    assert topic == 1 and math.isclose(f1, 0.8)  # topic 1 in A, B, C: TP=2, FP=1, FN=0

def test_best_topic_f1_no_positives():
    assert best_topic_f1(post_topics(OCC), CODES, "AI") == (None, None)

def test_match_quote_inside_and_spanning_sentences():
    assert match_quote("FLEKSTID", ["Pædagog", "Vi har flekstid."]) == [1]
    assert match_quote("fleksibel arbejdstid", ["Du får fleksibel", "arbejdstid og frihed."]) == []
    assert match_quote("Du får fleksibel arbejdstid og frihed.", ["Du får fleksibel", "arbejdstid og frihed."]) == [0, 1]

def test_match_quote_spanning_sentences():
    assert match_quote("Vi har flekstid. God løn.", ["Vi har flekstid.", "God løn.", "Kantine."]) == [0, 1]

def test_evidence_shares():
    shares = evidence_shares(OCC, CODES, ["flex", "AI"])
    assert shares == {"evidence_in_outliers_share": 0.0, "evidence_unmatched_share": 0.5}

def test_topic_diversity():
    assert topic_diversity([["a", "b"], ["b", "c"]], n=2) == 0.75
```

A quote counts as matched when `match_quote` returns at least one position. It counts as "in outliers" when all of its matched sentences are in topic -1. In the fixture, A's quote matches, and B's quote "fleksibel arbejdstid" crosses a sentence break without containing a whole sentence, so it is unmatched: 1 of 2 unmatched, 0 of 1 matched quotes in outliers. Only items with `present == 1` and a non-empty evidence quote count.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the helpers in `score.py`**

- `best_topic_f1` only considers coded posts (it joins `post_topics` to `codes` on `post_id`). For each topic, TP is the number of coded posts with the topic and `item == 1`, FP the number with the topic and `item == 0`, FN is `positives - TP`, and `F1 = 2TP / (2TP + FP + FN)`. It returns the topic with the highest F1, or `(None, None)` when there are no positives or no topics.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_score.py -v`
Expected: 6 passed.

- [ ] **Step 5: Implement `__main__` and the stage**

- `conf/score/default.yaml` holds `labels_path: data/llm_topics/claude-opus-5-5_low_prompt-3678b2ac/results.parquet` and `n_top_words: 10`.
- Add `- score: default` to the defaults in `conf/config.yaml`, and mirror it in `params.yaml`.
- `__main__` steps:
  1. Load the occurrences, the codes, and `BERTopic.load(models/bertopic_model)` (topic words only).
  2. Per item, compute `best_topic_f1`. Record `f1_<item>` and `best_topic_<item>`, and `mean_best_topic_f1` over the items that are not None.
  3. Add `evidence_shares`, the outlier share (per unique sentence and per occurrence), the topic count, and `topic_diversity`.
  4. Write `score.json`.
  5. Build `topics_overview.csv`: topic, size, top 10 words joined with ", ", 3 representative sentences (`topic_model.get_representative_docs(topic)[:3]`, falling back to 3 random sentences from that topic with `seed=1` if empty), and the best-matching item (the item whose `best_topic` is this topic, else empty).
- `dvc.yaml` gets a new `score` stage:
  - `cmd: python src/job_post_topic_modelling/score.py`
  - deps: `output/predicted_topics_train.parquet`, `models/bertopic_model/topics.json`, `src/job_post_topic_modelling/score.py`, and the labels folder `data/llm_topics/claude-opus-5-5_low_prompt-3678b2ac`
  - params: `score`
  - metrics: `output/metrics/score.json` (`cache: false`)
  - outs: `output/topics_overview.csv`

- [ ] **Step 6: Track the pilot codes with DVC**

1. Copy `../job-post-topic-modelling/data/llm_topics/claude-opus-5-5_low_prompt-3678b2ac/` into `data/llm_topics/`.
2. In `data/.gitignore`, replace `/llm_topics/` with `/llm_topics/*`, `!/llm_topics/*.dvc` and `!/llm_topics/.gitignore`.
3. Run `dvc add data/llm_topics/claude-opus-5-5_low_prompt-3678b2ac`.

Expected: a `.dvc` file is created. The configured cache is a STAR path. If `dvc add` fails locally because of it, stop and report to Adam rather than changing `.dvc/config`.

- [ ] **Step 7: Commit**

```bash
git add src/job_post_topic_modelling/score.py tests/test_score.py conf/score conf/config.yaml params.yaml dvc.yaml data/.gitignore data/llm_topics/*.dvc data/llm_topics/.gitignore
git commit -m "feat: score stage comparing topics with Claude codes"
```

---

### Task 9: Smoke run and comparison commands

**Files:**
- Modify: `common_runs.txt`

- [ ] **Step 1: Smoke run on 300 posts with model2vecdk**

```bash
uv run dvc exp run -S "prepare/star=local" -S "prepare.settings.nobs=300" -S "embed/model=model2vecdk" -S "embed.settings.device=cpu" prepare embed train score
```

Expected: all four stages finish. `output/metrics/score.json` exists and has `mean_best_topic_f1`. `prepare` prints drop counts and 20 headings. With ~6k unique sentences the floor gives `min_cluster_size` 10.

If it fails, debug with superpowers:systematic-debugging. Do not raise `nobs` above 300.

- [ ] **Step 2: Check the splitting success criteria on the smoke output**

Run a short polars check on `data/texts.parquet`.
Expected: under 0.25% of rows over 100 words, and none of the 25 most frequent sentences ends with `:`.

- [ ] **Step 3: Add the comparison queue to `common_runs.txt`**

```text
# Embedding comparison on Jobnet.xlsx (run only after the full Claude coding; tune min_cluster_size_share once on mpnet first)
dvc exp run --queue -S "prepare/star=local" -S "embed/model=sentence_transformers_big,model2vecdk,e5_large_instruct" --force-downstream
dvc exp run --queue -S "prepare/star=local" -S "embed/model=embeddinggemma" -S "embed.model.truncate_dim=null,256" --force-downstream
dvc exp run --queue -S "prepare/star=local" -S "embed/model=f2llm_330m,f2llm_160m" -S "embed.model.truncate_dim=null,256" --force-downstream
dvc exp show --only-changed --md
```

- [ ] **Step 4: Commit**

```bash
git add common_runs.txt
git commit -m "docs: commands for the embedding comparison"
```
