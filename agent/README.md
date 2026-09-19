# The agent side

Nothing in this folder is submitted. It runs on your machine, drives the API,
and is where your autoresearch loop lives.

For interactive use, run `../install-dryft.sh` (or `../install-dryft.ps1` on
Windows) to install `../bin/dryft`. The files here are small, dependency-free
Python building blocks for an automated research loop.

| File | What it does |
| --- | --- |
| `package.py` | Builds the archive from `engine/`, and refuses what the platform would refuse. |
| `client.py` | The submission API over the standard library. No dependencies. |
| `loop.py` | Evaluate an existing submission and print what it measured. |

The installed CLI already knows the event server. This Python loop does not, so
set both values before running it:

```sh
export DRYFT_API=https://dryft-user-testing.vercel.app   # no /api suffix
export DRYFT_TOKEN=dryft_pat_...                   # API tokens

python agent/loop.py SUBMISSION_ID                  # public samples, about two minutes
python agent/loop.py SUBMISSION_ID --mode official  # five samples, scored, ranked
```

Commit and push engine changes to your connected repository’s default branch
first. Copy the resulting submission ID from the website. The push starts its
configured evaluation; this script requests an additional run of those same bytes.

`loop.py` prints a row per workload with the measured time, the speedup over
native, and the time-to-first-token and time-per-output-token ratios. Those two
ratios are gates on an official run — above 1.10 and the workload fails — but a
public run only reports them, so read them before you promote a change.

`plan_next_edit` in `loop.py` is the part you write: given what previous
attempts measured, decide what to change about `engine/engine.py`. Keep a record
of every attempt. Only the three hidden workloads are scored, and the public
three will not always explain why a score moved.

Exit codes are `0` passed, `1` the request was refused, `2` the
run failed, `3` polling gave up. A poll that gives up has cancelled nothing:
keep the run id and look at it again rather than starting a second run.
