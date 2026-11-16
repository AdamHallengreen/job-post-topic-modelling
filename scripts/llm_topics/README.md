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

The model, the effort level and how many posts are sent at once are set at the top of the script
(`MODEL`, `EFFORT`, `POSTS_AT_A_TIME`).

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
* **Changing `prompt.txt`, `MODEL` or `EFFORT` starts a new folder**, so results from different setups
  never mix, and older ones are kept for comparison.

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

## Step 2: pilot (200 posts)

```powershell
uv run scripts/llm_topics/annotate_jobnet.py --n 200
```

### How long will the full run take on your plan?

Run `/usage` in Claude Code before and after the pilot. The difference is what 200 posts cost. Example:
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

Edit `prompt.txt` (or `MODEL` / `EFFORT` at the top of the script) and run the same pilot again. It goes
to a new folder, and because `--n 200` always picks the same posts, you can compare the two results post
by post.

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
| Input tokens far above a few thousand per post | Something extra is sent with every post; ask for help before the full run |
