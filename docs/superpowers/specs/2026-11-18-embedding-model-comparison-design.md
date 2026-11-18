# Comparing embedding models for BERTopic

## Intent

The topics should describe **traits of the job**: amenities (flexibility, hours, benefits) and skills.
Boilerplate (how to apply, company presentation, equal-opportunity text) may form its own topics but
is not what we are after. We want to know which embedding model gives the best trait topics on Danish
job posts, at a cost that still works for ~2M posts (~56M sentences) on STAR.

This round covers standard BERTopic only. Zero-shot and guided topic modeling come later, as new train
settings on the same pipeline.

**Readability is a hard requirement.** Adam must be able to read and follow every step, especially
the unique-sentence logic. Prefer plain, named steps over clever one-liners.

## Goals

- Five embedding models, each described by one small config file, selectable with
  `dvc exp run -S embed/model=<name>`.
- Better sentence splitting and cleaning in `prepare.py`.
- Each unique sentence is embedded once; every copy gets its sentence's topic.
- A new light `score` stage that compares topics with the Claude codes and writes metrics, so
  `dvc exp show` compares the models.
- Code that runs both locally (`data/Jobnet.xlsx`, ~10.5k posts) and on STAR (offline, 2M posts).

## Non-goals

- Zero-shot and guided BERTopic (next round).
- Changing `evaluate.py` beyond keeping it working (it needs STAR click data and a local LLM).
- Topic coherence scores that need new heavy dependencies (e.g. gensim).
- Choosing the winning model: the pipeline produces the numbers, Adam decides.

## Anti-goals

- Hidden magic in the dedup step (index juggling that is hard to follow).
- Different clustering settings per model: that would make the comparison unfair.
- Running the full pipeline on all local data before Adam says so.

## Approaches

- **A. Extend the existing DVC pipeline (chosen).** One config per model in `conf/embed/model/`, a new
  `score` stage, one `dvc exp` per model. Same code path locally and on STAR, DVC caches embeddings,
  and zero-shot/guided later become train settings.
- B. A standalone script that loops over the five models. Easy locally, but duplicates
  prepare/embed/train, does not scale to STAR, and the two paths would drift.

Also decided: **train on unique sentences** and map topics back to all copies. Stops repeated lines
like "Vi tilbyder:" (1,376 copies) from forming dense fake clusters, and trains faster. Trade-off:
topic sizes and c-TF-IDF words count each sentence once, not by how often it appears.

## What exists today (relevant parts)

- `prepare.py`: loads Excel (local) or pre-sectioned parquet (STAR), filters to Danish with Lingua,
  strips HTML, splits on blank lines/`•`/`*`, then NLTK `sent_tokenize`, drops "sensitive" sentences
  (dates, phones, e-mails, URLs). Writes `data/texts.parquet` (`label`, `text`).
- `embed.py`: one hard-coded model, `get_embedding_model_name` always returns a STAR path and only
  knows two models. Writes `.npy` shards row-aligned with `texts.parquet`.
- `train.py` / `predict.py`: assume embeddings are row-aligned with `texts.parquet`; extract the post
  ID with regex `^(\d+)_s` (STAR labels). **Locally the IDs are UUIDs, so `ann_id` is null today.**
- `load_excel` uses a global `par` (only works when run as `__main__`).

Measured on `Jobnet.xlsx` with the current splitter: 10,556 posts → 295,628 sentences (28/post),
32% of rows are exact duplicates, 10% are ≤3 words (mostly headings), a few run to 357 words because
lists separated by single line breaks are glued together, and phone numbers like "3634 4990" slip
through the sensitive filter.

## Design

### 1. Model configs (`conf/embed/model/*.yaml`)

Every config has the same fields:

```yaml
embedding_model: "google/embeddinggemma-300m"   # HF id; on STAR also the folder name under model_dir
prompt: "task: clustering | query: "            # null = no prompt
truncate_dim: null                               # null = full size; Matryoshka models only
max_seq_length: 256
dtype: auto                                      # auto | float32 | bfloat16
use_model2vec: false                             # existing option: distil to a static model
pca_dims: 768                                    # only used with use_model2vec
```

| Config file | Model | Full dims | Prompt | truncate_dim | max_seq_length | dtype |
|---|---|---|---|---|---|---|
| `sentence_transformers_big` (existing) | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | none | null | 128 | auto |
| `f2llm_330m` | `codefuse-ai/F2LLM-v2-330M` | 896 | none | 256 | 256 | bfloat16 |
| `f2llm_160m` | `codefuse-ai/F2LLM-v2-160M` | check card | none | 256 | 256 | bfloat16 |
| `embeddinggemma` | `google/embeddinggemma-300m` | 768 | `task: clustering \| query: ` | null | 256 | auto (never float16) |
| `e5_large_instruct` | `intfloat/multilingual-e5-large-instruct` | 1024 | `Instruct: Identify the topic of this sentence from a Danish job post\nQuery: ` | null | 256 | auto |
| `model2vecdk` | `andersborges/model2vecdk` | 256 (verify) | none | null | n/a | float32 |

Notes:
- `dtype: auto` = bfloat16 on GPU, float32 on CPU (bfloat16 is slow on many CPUs). Passed as
  `model_kwargs={"torch_dtype": ...}`. Raise a clear error if float16 is ever requested for Gemma.
- `max_seq_length: 256` is our call: sentences are short (median 13 words), and a shared cap keeps
  speed comparable. mpnet keeps its own limit of 128.
- The e5 instruction text and the choice of "no prompt" for F2LLM are our calls (the F2LLM card says a
  prompt is optional for clustering).
- Matryoshka sizes are swept with `-S embed.model.truncate_dim=...`: run Gemma at 768 and 256, F2LLM
  at 896 and 256.
- `model2vecdk` loads directly with `SentenceTransformer("andersborges/model2vecdk")`, not through the
  `use_model2vec` distillation path.
- Existing configs (`sentence_transformers_small`, `intfloat`, `GritLM`) get the new fields with
  values that keep their current behaviour.
- Gemma needs the licence accepted on Hugging Face and an `HF_TOKEN` locally.

### 2. Loading a model (`embed.py`)

- New setting `embed.settings.model_dir`: `null` = download from Hugging Face (local);
  a path = load `model_dir / embedding_model` offline (STAR). Replaces the STAR-only
  `get_embedding_model_name` logic; offline env vars are set only when `model_dir` is set.
- One function `load_embedding_model(model_par, settings)` builds the `SentenceTransformer` with
  device, dtype, `truncate_dim` and `max_seq_length`. `train.py` and `predict.py` reuse it (on CPU).

### 3. Normalization

Always L2-normalize, **after** truncation, in our own code (a short, named `normalize` step), not
relying on library flags. Reasons: UMAP-cosine does not care, but the PCA+KMeans path used for full
runs does, and truncated Matryoshka vectors must be renormalized. Cosine similarity then equals dot
product, which zero-shot/guided can rely on later.

### 4. Sentence splitting and cleaning (`prepare.py`)

1. Before stripping HTML, turn line-break and list tags (`<br>`, `<li>`, `</p>`, `</div>`) into `\n`.
2. Split paragraphs on any line break (not only blank lines), plus `•` and `*` as today, then NLTK
   `sent_tokenize(language="danish")`.
3. Drop headings: a sentence is a heading if it ends with `:`, or if it has ≤4 words and does not end
   with `.` or `!`. Keeps "Vi har flekstid." and "fastansættelse."; drops "Vi tilbyder:",
   "Hvem er du?", "Om opgaven". Rule is our call; print 20 random dropped examples in the log so it can
   be checked.
4. Sensitive filter: add phone formats `dddd dddd` and `+45 ...`.
5. Keep the Danish-language filter as is.
6. Write a `post_id` column explicitly instead of parsing it from `label` later. Locally: the Excel
   `ID`. On STAR (contents of the file not known): use an `ann_id` column if
   `jobads_sections_clean.parquet` has one, otherwise extract it from `section_id` with today's
   regex `^(\d+)_s`. Fix `load_excel` to take `par` as an argument.
7. **Job title as the first sentence of each post** (Adam's decision). Claude coded `Title + Text`,
   so quotes from the title must be matchable, and titles carry occupation and skills. Locally read
   the Excel `Title` column; on STAR use a title column if one exists, otherwise skip (log it).
8. **STAR runs through the same code** (Adam's decision): the line-break split, heading rule and
   sensitive filter also run on STAR sections.

Success check for splitting (on the local data): share of sentences over 100 words clearly below
today's 0.25%, and headings gone from the 25 most frequent sentences.

### 5. Unique sentences

Done in `prepare.py`, in plain named steps, so later stages never juggle indices:

- `data/texts.parquet`: one row per sentence occurrence: `post_id`, `label`, `text`, `sentence_id`.
- `data/sentences.parquet`: one row per unique sentence: `sentence_id`, `text`, `n_occurrences`,
  in first-seen order. `sentence_id` is simply the row number in this file.

Then:
- `embed.py` embeds `sentences.parquet` only. Shard row *i* = `sentence_id` *i*.
- `train.py` fits BERTopic on the first `train.settings.nobs` unique sentences, saves topics per
  `sentence_id`, then joins them onto `texts.parquet` by `sentence_id` to get per-occurrence and
  per-post outputs.
- `predict.py` keeps today's behaviour: it scores **every** sentence (training ones too, marked with
  `training_data`) with the current `full_p_dist` path and wide `topic_i` probability columns. The
  only change is that it computes this per unique sentence and joins back by `sentence_id`.
  (Our call, so `evaluate.py` keeps working on STAR.)
- Output files keep their current names and columns (`ann_id` = `post_id`) so `evaluate.py` still
  works on STAR.

### 6. Train improvements (`train.py`, `conf/train/`)

**Relative cluster thresholds**, so one config works on `Jobnet.xlsx` and on STAR:
- `conf/train/clustering/HDBSCAN.yaml` replaces `min_cluster_size: 3000` / `min_samples: 200` with
  `min_cluster_size_share: 0.0005` (share of training sentences) and `min_samples_ratio: 0.1`
  (fraction of `min_cluster_size`). `train.py` turns these into integers in one small, named
  function (with a floor of e.g. 10), prints them, and writes them to `output/metrics/train.json`.
  Gives ~4,000 on 8M sentences and ~90 on ~180k local unique sentences.
- Stay absolute on purpose: UMAP `n_neighbors` (local neighbourhood size, not data size), KMeans
  `n_clusters` (wanted number of topics), `nobs` (memory cap).
- Tuning (our call): **do not tune on the Claude F1** (would favour mpnet). Set
  `min_cluster_size_share` once so mpnet gives a sensible topic count (roughly 100–300) and a moderate
  outlier share, then keep it fixed for all models. This run uses all local posts, so it waits for
  Adam's go-ahead.

**Other changes:**
1. **Random training sample** with a fixed seed (`train.settings.seed`) instead of `head(nobs)`, so
   STAR training is not just the earliest posts. Store which `sentence_id`s were used for training;
   `predict.py` marks `training_data` from that list instead of `row_nr < nobs`.
2. **Same embedding model in train/predict as in embed**: built with the shared
   `load_embedding_model` (same `truncate_dim`, dtype, prompt), on CPU. Needed so seed-word
   embeddings (guided modeling, next round) have the same size as the sentence embeddings.
3. **Fixed seeds by default**: UMAP and KMeans `random_state` = `train.settings.seed` (42).
4. **One thread setting** `train.settings.n_jobs` used for UMAP `n_jobs` and HDBSCAN
   `core_dist_n_jobs`, matching the thread limits in `dvc.yaml` (today 16 vs 8 vs 8).
5. **Clean-up**: remove unused `rescale`, `UnsupportedAnchorTypeError`, and unused arguments
   (`embeddings` in `get_dimensionality_reduction_model`, `stop_words` in `load_model_objects`);
   pickle files written with `with open(...)`.
6. **Optional outlier reassignment**: `train.settings.reduce_outliers` (`null` = off, default; or a
   BERTopic strategy such as `embeddings`, with `reduce_outliers_threshold`). Applied after fitting,
   followed by `update_topics`. Lets `score` show whether it helps; off keeps junk out of topics.

Noted, not changed: the full-scale route uses PCA to 5 dimensions before KMeans, which keeps little
of a 768-dim vector; 20–50 dimensions is worth testing later.

### 7. `score` stage (new `score.py`)

Inputs: `output/predicted_topics_train.parquet`, `data/texts.parquet`, topic words from the saved
model, and Claude codes from `score.labels_path` (a `results.parquet` under `data/llm_topics/...`;
join on `ID` = `post_id`; only posts with `valid == 1`). Writes `output/metrics/score.json` (a DVC
metric) and `output/topics_overview.csv`.

Metrics:
1. **Best-topic F1 per Claude item.** For each of the 15 items, and each topic: a post "has" the topic
   if any of its sentences is in it. F1 of that against Claude's `present`. Report the best topic and
   its F1 per item, plus the mean over items as the headline number.
2. **Evidence in outliers.** Match each Claude evidence quote to the post's sentence(s) that contain
   it (whitespace- and case-insensitive). Report the share of matched quotes in topic -1, and the
   share of quotes that could not be matched (a sanity check on splitting).
3. **Outlier share and topic count**, counted both per unique sentence and per occurrence.
4. **Topic diversity**: share of unique words among all topics' top-10 words.
5. **Cost** (written by `embed.py` to `output/metrics/embed.json`): sentences per second, vector
   size, bytes per vector, and projected storage for 56M sentences. Speed is measured on whatever
   machine runs it, so it compares models with each other; a timing run on STAR is a later step
   when the models are copied there (our call).

`topics_overview.csv`: per topic, size, top 10 words, 3 representative sentences, and the Claude
item it best matches. This is what Adam reads to judge skills, which Claude does not code.

**The Claude codes are tracked with DVC** (Adam's decision): `dvc add` the chosen results folder under
`data/llm_topics/`, and list it as a dependency of `score`, so queued experiments see it and scoring
reruns when more posts are coded. Until the full Claude run exists, use the 100-post pilot
(`claude-opus-5-5_low_prompt-3678b2ac`, currently in the main checkout; copy it into this worktree)
to develop and test the stage.

### 8. Running the comparison

After the full Claude run, queue one experiment per model (and Matryoshka size) with
`prepare/star=local`, then `dvc exp show`. The queue commands
go in `common_runs.txt`.

## Testing

- `pytest` unit tests on small hand-made inputs: splitting (line-break lists, HTML list tags),
  heading rule, phone filter, dedup mapping (`texts` ↔ `sentences` round-trip), normalize after
  truncate, and the F1/quote-matching functions in `score.py`.
- A smoke run on a few hundred posts (`prepare/settings=small`) with `model2vecdk` to check the stages
  connect. **No runs on all local posts until Adam says so.**

## Left to the builder

- Exact function and file layout within the stages, as long as it stays easy to read.
- How shards are written for `sentences.parquet` (keep the current shard format).
- Whether `n_occurrences` is used beyond reporting.
