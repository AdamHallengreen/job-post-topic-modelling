# Coding Jobnet posts with Claude

`annotate_jobnet.py` sends each post in `data/Jobnet.xlsx` (10,553 posts with text) to Claude, together
with the instructions in `prompt.txt`. For each of the 15 items, Claude returns a short quote from the post
(`evidence`) and a 0/1 decision (`present`). It also says whether the text is a real job post at all.

There are only two options:

| Command | What it does |
|---|---|
| `uv run scripts/llm_topics/annotate_jobnet.py --n 200` | Pilot: code a random sample of 200 posts (always the same 200) |
| `uv run scripts/llm_topics/annotate_jobnet.py` | Code all posts |
| add `--api` | Pay per token with `ANTHROPIC_API_KEY` instead of using your Claude subscription |
| add `--model claude-sonnet-5-5` | Use another model than the default (`MODEL` at the top of the script) |
| add `--effort high` | Use another effort level than the default (`EFFORT` at the top of the script): `low`, `medium`, `high`, `xhigh` or `max` |

How many posts are sent at once is set at the top of the script (`POSTS_AT_A_TIME`). To compare models
and effort levels on a pilot, use the DVC experiments in
[Comparing models with DVC](#comparing-models-with-dvc).

## How saving works

Results go to a folder named after the model, effort and prompt version, for example
`data/llm_topics/claude-haiku-5-5_low_prompt-1a2b3c4d/`:

| File | What |
|---|---|
| `results.jsonl` | One line per coded post, written the moment the answer arrives |
| `errors.jsonl` | One line per failed attempt (safe to delete) |
| `results.parquet` | The table you analyse: one row per coded post, rebuilt at the end of every run |

This has three consequences:

* **Running the same command again continues where the last run stopped.** Posts that are already
  coded are skipped, and posts that failed are tried again.
* **The pilot counts toward the full run.** After a 200-post pilot, the full run only codes the rest.
* **Changing `prompt.txt`, the model or the effort starts a new folder**, so results from different
  setups never mix, and older ones are kept for comparison.

## Before the first run

**Subscription (the default).** Log in to Claude Code once with your Pro/Max account:

```powershell
claude          # not installed? irm https://claude.ai/install.ps1 | iex
# inside Claude Code: /login and pick your Claude account, then /exit
```

The script starts Claude Code once per post with that login. Your plugins, hooks, connectors and
memory are switched off for these sessions, so Claude only sees `prompt.txt` and the post.
`ANTHROPIC_API_KEY` is ignored in this mode, so you can't be billed on the API by accident. If you've
turned on paid extra usage for your plan on claude.ai, usage past your limit is billed; turn it off if
you'd rather the run stopped.

**API (`--api`).** Set a key in the shell first: `$env:ANTHROPIC_API_KEY = "sk-ant-..."`.

The first run takes a minute while uv installs the script's dependencies (Claude Code alone is about
160 MB).

## Step 1: smoke test (5 posts)

```powershell
uv run scripts/llm_topics/annotate_jobnet.py --n 5
```

You should see `5/5 posts, 0 failed` and a table (explained below). Check **`Mean tokens per post`**:
the input should be a few thousand tokens (the prompt plus one post). Much more means something extra
is being sent with every post.

## Step 2: pilot (100 posts)

```powershell
uv run scripts/llm_topics/annotate_jobnet.py --n 100
```

### How long will the full run take on your plan?


Run `/usage` in Claude Code before and after the pilot. The difference is what 100 posts cost. Example:
if they use 8% of the 5-hour limit, one 5-hour window covers about 2,500 posts, so all 10,553 posts take
about 4–5 windows. Check the weekly percentage the same way.

### Check the quality

Every run ends with a table (made-up numbers):

```
item              share_coded_1  quotes_found_in_post
Weekends          0.31           0.98
wfhOptions        0.12           1.00
...
```

* `share_coded_1`: share of posts coded 1. Does it match what you'd expect for Danish job ads?
* `quotes_found_in_post`: of the quotes Claude gave, the share that really appear in the post (ignoring
  case and spacing). Well below 1.00 means Claude paraphrases instead of quoting. In `results.parquet`,
  the rows with `<item>_quote_ok == false` show which quotes.

Then read 20–30 posts yourself and compare them with the coding. This writes them to a CSV that opens
correctly in Excel (put in your own folder name):

```powershell
uv run python -c "import polars as pl; pl.read_parquet('data/llm_topics/<folder>/results.parquet').head(30).write_csv('pilot_review.csv', include_bom=True)"
```

The `row` column is the post's position in `Jobnet.xlsx`. In Excel it's on row `row + 2`, because the
sheet starts with a header row.

### Try a different prompt, model or effort

Edit `prompt.txt` (or pass `--model` or `--effort`) and run the same pilot again. It goes to a new
folder, and because `--n 200` always picks the same posts, you can compare the two results post by
post. To compare several setups in an organized way, see the next section.

## Comparing models with DVC

The DVC stage `llm_pilot` in `dvc.yaml` codes a pilot of 100 posts. Its three settings are in
`conf/llm_pilot/default.yaml`: `model`, `effort` and `n_posts`. Each DVC experiment runs the stage with
one model and effort level and records them together with summary numbers for those 100 posts
(`output/metrics/llm_pilot.json`). Afterwards `dvc exp show` puts the experiments side by side, and
`compare_pilots.py` shows post by post where they agree.

All experiments code the same 100 posts. Their answers are saved in each setup's own folder in
`data/llm_topics/`, the same folders a normal run uses, so a later full run with the winning setup
skips these 100 posts.

### Once, before the first experiment

The repo's DVC cache setting points at the server (`/data/projects/...`). Point it to a local folder on
this computer. The setting goes in `.dvc/config.local`, which isn't committed, so the server keeps its
own setting:

```powershell
uv run dvc config --local cache.dir .dvc/cache
```

Experiments start from your last commit plus any uncommitted changes in the workspace. Committing the
setup first makes it clear later which code each experiment ran:

```powershell
git add conf/config.yaml conf/llm_pilot params.yaml dvc.yaml scripts/llm_topics
git commit -m "Add llm_pilot stage for comparing Claude models"
```

### Run the pilots

Run them one at a time. Each takes roughly 5–15 minutes. Run `/usage` in Claude Code before and after
each one to see how much of your limits it uses. Opus uses them up much faster than Haiku, and higher
effort makes Claude write more: on the first pilot, Haiku wrote about 1,500 tokens per post at `low`
and 2,700 at `high`.

```powershell
uv run dvc exp run llm_pilot -n pilot-opus-low   -S llm_pilot.model=claude-opus-5-5  -S llm_pilot.effort=low
uv run dvc exp run llm_pilot -n pilot-haiku-low  -S llm_pilot.model=claude-haiku-5-5 -S llm_pilot.effort=low
uv run dvc exp run llm_pilot -n pilot-haiku-high -S llm_pilot.model=claude-haiku-5-5 -S llm_pilot.effort=high
```

* `llm_pilot` tells DVC to run only this stage, not the BERTopic pipeline.
* `-n` names the experiment. `-S` sets a parameter for this experiment only. Leave out
  `-S llm_pilot.effort=...` to use the default in `conf/llm_pilot/default.yaml`.
* Don't use `--queue` here. Queued experiments run in a temporary copy of the repo that doesn't have
  `data/Jobnet.xlsx`, and several at once would use up your limits together.
* If one stops at the usage limit, the experiment isn't saved, but the posts it coded are. Run the same
  command after the limit resets and it finishes the rest.

### Compare the results

The summary numbers per experiment:

```powershell
uv run dvc exp show --only-changed
```

There is one row per experiment, with the model, effort, `quotes_found_in_post`, `mean_input_tokens`,
`mean_output_tokens`, `not_job_post` and `share_coded_1` for each item. `--only-changed` hides every
column that's the same in all experiments, such as all the BERTopic settings.

Then where the setups agree, post by post:

```powershell
uv run scripts/llm_topics/compare_pilots.py
```

It prints one row per item:

* `share_1 <model>_<effort>`: share of the 100 posts each setup coded 1.
* `all_agree`: share of posts where all setups gave the same code. For rare items this is high anyway,
  because most codes are 0.
* `all_agree_when_any_1`: of the posts where at least one setup said 1, the share where all of them did.
  This is the number to watch for rare items.

It also writes `data/llm_topics/pilot_disagreements.csv` (opens in Excel), with one row per post and
item where the setups disagree, each setup's code and quote side by side. Reading 20–30 of these tells
you which one is right more often. If the cheaper setup agrees with the most capable one on almost
everything, use the cheaper one for the full run.

### Afterwards

After the runs, your workspace holds the settings of the last experiment. To go on with one of them,
for example Haiku at low effort:

```powershell
uv run dvc exp apply pilot-haiku-low
git add params.yaml dvc.lock output/metrics/llm_pilot.json
git commit -m "Pilot: pick claude-haiku-5-5 at low effort"
```

Then run all posts with that setup:
`uv run scripts/llm_topics/annotate_jobnet.py --model claude-haiku-5-5 --effort low`, or set `MODEL` and
`EFFORT` at the top of the script.

## Step 3: all posts

```powershell
uv run scripts/llm_topics/annotate_jobnet.py
```

When your usage limit runs out, the script stops with `Stopping: every post in the last group failed`
and tells you how many posts are left. Run the same command again after the limit resets. Nothing is
lost.

To keep going automatically, run this loop. It retries once an hour until every post is coded:

```powershell
do {
    uv run scripts/llm_topics/annotate_jobnet.py
    if ($LASTEXITCODE -ne 0) { Start-Sleep -Seconds 3600 }
} while ($LASTEXITCODE -ne 0)
```

Keep the computer plugged in and awake while it runs. Ctrl+C stops it at any time.

To finish faster, you can code the remaining posts on the API instead (`--api`). They go into the same
folder.

## The results table

`results.parquet` has one row per coded post:

* `row`, `ID`, `Title`
* for each item: `<item>` (0/1), `<item>_evidence` (the quote) and `<item>_quote_ok` (whether the quote
  appears in the post; empty when there is no quote)
* `valid` (1 = a real job post) and `valid_reason`
* `input_tokens`, `output_tokens`

## When something goes wrong

| What you see | What to do |
|---|---|
| Every post fails with a login or authentication error | Run `claude`, then `/login` with your Pro/Max account |
| It stops straight away with a usage-limit error | Your limit is used up; check `/usage` and wait for the reset |
| A few posts fail | Run the same command again; they are retried. The reasons are in `errors.jsonl` |
| `dvc exp run` stops with `failed to rename lockfile to '.git/HEAD': Access is denied` | Another program, often VS Code's Git view, briefly locked a Git file. Run the same command again |
| Input tokens far above a few thousand per post | Something extra is sent with every post; ask for help before the full run |
