"""Server entry point. Never writes into the original paper_final_v1 results."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "all", "matched", "grouped", "benchmark", "summarize", "pack-results", "_worker"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_supplement_v2.json")
    parser.add_argument("--old-root", type=Path, default=Path("/root/autodl-tmp/iotids_paper_final_v1"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--legacy-results", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_supplement_v2")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--stage", choices=("train", "evaluate", "benchmark"))
    parser.add_argument("--protocol", choices=("legacy", "grouped"))
    parser.add_argument("--scenario")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    args.old_root, args.output, args.config = args.old_root.resolve(), args.output.resolve(), args.config.resolve()
    args.dataset = (args.dataset or args.old_root / "datasets/Farm-Flow/Farm-Flows.csv").resolve()
    args.legacy_results = (args.legacy_results or args.old_root / "outputs/paper_final_v1").resolve()
    if args.output == args.legacy_results or args.output.is_relative_to(args.old_root):
        raise ValueError("Supplement output must be outside the original project")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from iotids_supplement.suite import preflight, suite_lock, execute, summarize, pack_results, worker
    if args.command == "_worker":
        if any(v is None for v in (args.stage, args.protocol, args.scenario, args.seed)):
            parser.error("Internal worker arguments missing")
        worker(ROOT, config, args.stage, args.protocol, args.scenario, args.seed,
               args.dataset, args.legacy_results, args.output, args.allow_cpu)
        return
    if args.command == "preflight":
        preflight(ROOT, config, args.old_root, args.dataset, args.legacy_results, args.allow_cpu)
        return
    with suite_lock(args.output):
        if args.command == "summarize":
            summarize(ROOT, config, args.output)
        elif args.command == "pack-results":
            pack_results(ROOT, config, args.output)
        else:
            # Preflight in a fresh process: TensorFlow must not hold GPU memory
            # in the parent while training workers run.
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "preflight",
                       "--config", str(args.config), "--old-root", str(args.old_root),
                       "--dataset", str(args.dataset), "--legacy-results", str(args.legacy_results),
                       "--output", str(args.output)]
            if args.allow_cpu:
                command.append("--allow-cpu")
            subprocess.run(command, check=True, cwd=ROOT)
            stages = ["matched", "grouped", "benchmark"] if args.command == "all" else [args.command]
            execute(ROOT, args, config, stages)
            pack_results(ROOT, config, args.output)


if __name__ == "__main__":
    main()
