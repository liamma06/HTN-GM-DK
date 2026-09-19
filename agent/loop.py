"""One turn of the autoresearch loop: package, submit, run, read the result.

    python agent/loop.py                  # public samples, about two minutes
    python agent/loop.py --mode official  # five samples, scored, ranked

This is the outer cycle only. The part that decides *what to change* about
``engine/engine.py`` is yours, and is the whole exercise; see `plan_next_edit`
at the bottom. Everything above it exists so that a proposed edit can be turned
into a measured number without a human in the loop.

Two habits worth keeping from the start. Record every attempt — the edit, the
archive digest, the run id, and the per-workload numbers — because the hidden
workloads are the only ones scored and the public three will not always show you
why a score moved. And never start a second run because the first was slow to
answer: a poll that times out has not cancelled anything.
"""

import argparse
import sys
from pathlib import Path

from client import ApiError, Dryft
from package import package

#: Time to first token and time per output token may not exceed this multiple
#: of native's. Official runs enforce it; public runs only report the ratios.
LATENCY_GATE = 1.10

ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"


def report(detail: dict) -> bool:
    """Print what the run measured. Returns True if it passed everything."""
    state = detail.get("state")
    result = detail.get("result") or {}
    shapes = result.get("shapes") or []

    print(f"run {detail.get('id')}: {state}")
    if result.get("score") is not None:
        print(f"score {result['score']:.1f}  (100 is native)")

    for shape in shapes:
        metrics = shape.get("modelMetrics") or {}
        columns = [f"{shape['id']:<10}", f"{shape.get('caseStatus', '?'):<13}"]
        if shape.get("metricMs") and metrics.get("referenceMs"):
            speedup = metrics["referenceMs"] / shape["metricMs"]
            columns.append(f"{shape['metricMs']:8.1f} ms  {speedup:5.2f}x native")
        for name, mine, native in (
            ("ttft", metrics.get("ttftMs"), metrics.get("referenceTtftMs")),
            ("tpot", metrics.get("tpotMs"), metrics.get("referenceTpotMs")),
        ):
            if mine and native:
                ratio = mine / native
                flag = "  OVER GATE" if ratio > LATENCY_GATE else ""
                columns.append(f"{name} {ratio:4.2f}x{flag}")
        if shape.get("tokensPerSecond"):
            columns.append(f"{shape['tokensPerSecond']:7.1f} tok/s")
        print("  " + "  ".join(columns))
        if shape.get("caseMessage"):
            print(f"    {shape['caseMessage']}")

    for label, value in (
        ("failure", result.get("failureMessage") or result.get("failureCode")),
        ("error", detail.get("errorMessage") or detail.get("errorCode")),
        ("not ranked", result.get("rankingReason")),
    ):
        if value:
            print(f"  {label}: {value}")

    return state == "succeeded" and all(
        shape.get("caseStatus") != "failed" for shape in shapes
    )


def attempt(client: Dryft, engine_dir: Path, mode: str, timeout: float) -> bool:
    archive = package(engine_dir)
    print(f"packaged {engine_dir} -> {len(archive)} bytes")

    submission_id = client.submit(archive)
    started = client.start_run(submission_id, mode=mode)
    run_id = started["id"]
    print(f"submission {submission_id}, {mode} run {run_id}; waiting")

    return report(client.wait(run_id, timeout=timeout))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ENGINE_DIR,
                        help="the folder whose contents are submitted")
    parser.add_argument("--mode", choices=("public", "official"), default="public",
                        help="public samples are feedback; official runs rank")
    parser.add_argument("--timeout", type=float, default=3000,
                        help="seconds to poll before giving up on the answer")
    arguments = parser.parse_args()

    try:
        client = Dryft()
        passed = attempt(client, arguments.engine, arguments.mode, arguments.timeout)
    except ValueError as problem:
        print(f"the archive was refused before upload: {problem}", file=sys.stderr)
        return 1
    except ApiError as problem:
        print(f"the API refused the request: {problem}", file=sys.stderr)
        return 1
    except TimeoutError as problem:
        print(f"{problem}; the run is still going, look it up rather than starting another",
              file=sys.stderr)
        return 3
    return 0 if passed else 2


def plan_next_edit(history: list[dict]) -> str:
    """Decide what to change about the engine next. This is the exercise.

    ``history`` is whatever you have chosen to keep from previous attempts:
    the edit, the archive digest, the run id, and the per-workload numbers.

    Somewhere to start, using the public workloads for feedback:

    1. Per-step overhead. The baseline pays full Transformers dispatch on every
       decode step. CUDA graphs, or a step that skips the model wrapper.
    2. Prefill, which is a large share of a short output's total time.
    3. A preallocated KV cache, instead of whatever the wrapper returns.
    4. Fused kernels for the small repeated pieces; see ``engine/kernels``.
    5. Speculative decoding with exact verification, once the rest is taken.

    Whatever you try, the output tokens must not change. Read AGENTS.md before
    reaching for anything that alters the arithmetic rather than reordering it.
    """
    raise NotImplementedError("this is the part you write")


if __name__ == "__main__":
    raise SystemExit(main())
