# Decode Qwen3 4B faster

One benchmark. Decode `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, on one H100, faster than
native Qwen — without changing a single token native Qwen would have produced.

`engine/engine.py` is native Qwen: Transformers, BF16, a greedy loop with a KV
cache. It is the baseline every score is measured against, so submitted
unchanged it targets a score of 100 (measured runs can vary slightly). Everything above 100 is yours to find.

Read the **Docs** page in the app before you optimize. It explains the
workloads, timing, output rule, and scoring. `QWEN_ENGINE_CONTRACT.md` is the
same guide for offline use.

## Start here

1. Select **Use this template → Create a new repository** on
   [Dryft-Kernels/starter](https://github.com/Dryft-Kernels/starter). Choose your
   account and make the new repository private if you want to keep your engine private.
2. Clone your new repository to your computer.
3. [Sign in to Dryft](https://dryft-user-testing.vercel.app/) with GitHub and
   create a team or join one with a six-character invite code from a team owner
   or admin. The code joins you immediately.
4. As a team owner or admin, open [Submissions](https://dryft-user-testing.vercel.app/bench), select
   **Connect a repository**, and grant the GitHub App access to your new repository.
   Set **Engine folder** to `engine`.
5. Push to your default branch to run the public samples, or use **Run now** in
   [Repositories](https://dryft-user-testing.vercel.app/repos) for the initial commit.
   Check the results before changing the engine. Public runs give feedback;
   request an official evaluation to appear on the leaderboard.

Ordinary members can run a repository after an owner or admin connects it.
If you cannot connect a repository, open Submissions and use **Download starter
archive** and **Submit an archive**. Uploading does not start a run: open the
new submission and select **Run sample cases**.

No local GPU or Python setup is needed for this workflow. Edit `engine/engine.py`
and push again to test an improvement. For the full guide, see
[Docs](https://dryft-user-testing.vercel.app/docs). For CLI submissions, follow
[Submitting](#submitting) below.

## Layout

| Path | Submitted | What it is |
| --- | --- | --- |
| `engine/engine.py` | yes | Your engine. Native Qwen until you replace it. |
| `engine/kernels/` | yes | Worked Triton example. Delete it or build on it. |
| `agent/` | **no** | Your autoresearch loop. Runs on your machine. |
| `bin/` | **no** | The installed Dryft CLI. |
| `requirements.txt` | no | The container's versions, for a local GPU. |
| `AGENTS.md` | no | The contract as rules, for a coding agent. |

Only `engine/` is submitted. Keep your agent, notes, and credentials outside it.

## The format

The format is fixed:

- the archive root holds `engine.py`;
- `engine.py` exports `class Engine`;
- `Engine` has exactly two methods.

Dryft names repository submissions from the repository and your team’s stable
slot, such as `fast-qwen #7`.

```python
class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yield a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence in input_ids has the
        same length. Do not stop at end-of-sequence tokens.
        """
```

Put any Python or Triton files your engine imports beside `engine.py`. The run
container has no network, so your engine cannot install or download anything.

## Submitting

No Python setup is needed. From this folder, run the installer for your system.
It downloads the public CLI, checks it, and puts it in `bin/`:

```sh
./install-dryft.sh

# Windows PowerShell:
# .\install-dryft.ps1
```

Connect this repository in Dryft, set its engine folder to `engine`, and enable
**Auto-run on push**. New submissions come only from default-branch pushes.
Create a token under [API tokens](https://dryft-user-testing.vercel.app/tokens)
to inspect results or rerun an existing submission:

```sh
export DRYFT_TOKEN='dryft_pat_...'

./bin/dryft doctor
./bin/dryft validate engine
git add engine
git commit -m "Update engine"
git push origin HEAD
```

The push starts the evaluation selected in **Evaluation on push**. Copy an
existing submission ID from the website to rerun it with
`./bin/dryft run <submission-id> --mode public --wait 3000`. The CLI already
knows the event server, so most people only need `DRYFT_TOKEN`.

Useful follow-up commands:

```sh
./bin/dryft submissions                  # latest 25 submissions
./bin/dryft runs                         # latest 25 runs
./bin/dryft logs <run-id> --follow
./bin/dryft result <run-id> --wait --timeout 3000
./bin/dryft cancel <run-id> --reason 'superseded'
```

Set `DRYFT_API` only for local development, staging, or a self-hosted server.

The CLI accepts the engine folder, `engine.py`, or an existing `.tar.gz`. If
you need to package by hand, run this from the starter folder:

```sh
cd engine && tar -czf ../submission.tar.gz engine.py kernels
```

Name the files explicitly. `tar -C engine .` writes paths such as
`./engine.py`, which the platform rejects. Use `python agent/loop.py SUBMISSION_ID` to evaluate an existing submission;
push a new commit whenever the engine changes.

Connect this repository in Dryft and set its engine folder to
`engine`. Each default-branch push can run the public samples automatically.
Public runs are for feedback; official runs can update the leaderboard.

At most 2 MiB compressed, 16 MiB expanded, 200 files. Supported extensions are
`.py .pyi .yaml .yml .json .toml .txt .md .cfg .ini`. No weights, no
credentials, no compiled binaries, no Docker images, no agent.

## The rule you cannot bend

Every token you emit is replayed through native Qwen on your own prefix and
must be the greedy choice there, or within 2.0 logits of it. The margin exists
because BF16 arithmetic in a different order flips genuine near-ties; it does
not admit approximations. A quantized model, a pruned cache or an approximate
attention shifts logits by whole units on some prompt, and one failing position
fails the workload.

Faster is yours: KV cache layout and paging, CUDA graphs over decode steps,
fused or custom Triton kernels, chunked or overlapped prefill, speculative
decoding with exact verification. The answer is not yours to change.
