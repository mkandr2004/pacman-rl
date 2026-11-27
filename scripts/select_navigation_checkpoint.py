"""Compare navigation on separate validation seeds before updating one model."""
import argparse
import json
import os
import shutil
from pathlib import Path
from statistics import mean

from scripts.audit_policy import audit
from scripts.resume_ppo import digest


def metrics(results):
    return dict(wins=sum(r["victory"] for r in results),
        collection=mean(r["collection_fraction"] for r in results),
        return_count=mean(r["two_step_returns"] for r in results),
        return_ratio=mean(r["two_step_returns"] / max(1, r["steps"]-2) for r in results),
        safe=all(r["outside_frames"] == 0 and r["borders_broken"] == 0 for r in results))


def should_promote(baseline, proposed):
    return (proposed["safe"] and proposed["wins"] >= baseline["wins"]
        and proposed["collection"] >= baseline["collection"]
        and (proposed["wins"] > baseline["wins"] or (
            proposed["return_ratio"] <= baseline["return_ratio"]
            and proposed["return_count"] <= baseline["return_count"]
            and any(proposed[k] != baseline[k] for k in ("collection", "return_ratio", "return_count")))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", help="Original backup used as the baseline")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, candidate, root = Path(args.model).resolve(), Path(args.candidate).resolve(), Path(args.output)
    reference = Path(args.reference).resolve() if args.reference else source
    root.mkdir(parents=True, exist_ok=False)
    source_hash = digest(source)
    backup = root / "original_model.zip"
    shutil.copy2(source, backup)
    if digest(backup) != source_hash:
        raise RuntimeError("Backup verification failed")
    protocol = dict(validation_seeds=list(range(9000, 9005)),
        test_seeds=[1000, 7000, 7001, 7002, 7003],
        criterion="safe; no fewer wins/collection; more wins OR no more absolute/relative two-step returns; strict improvement required",
        source=str(source), candidate=str(candidate), source_sha256=source_hash)
    protocol["reference"] = str(reference)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    baseline = metrics(audit(reference, protocol["validation_seeds"], root / "baseline_validation"))
    proposed = metrics(audit(candidate, protocol["validation_seeds"], root / "candidate_validation"))
    promote = should_promote(baseline, proposed)
    # Test is reported after the validation decision and cannot change selection.
    selected = candidate if promote else reference
    results = audit(selected, protocol["test_seeds"], root / "test")
    if digest(selected) != digest(source):
        if digest(source) != source_hash:
            raise RuntimeError("Source changed; refusing overwrite")
        staging = source.with_name(f"{source.stem}.navigation_pending.zip")
        if staging.exists():
            raise FileExistsError(staging)
        shutil.copy2(selected, staging)
        os.replace(staging, source)
    report = dict(protocol=protocol, baseline=baseline, proposed=proposed,
        promoted=promote, tests=results, final_sha256=digest(source),
        backup_verified=digest(backup) == source_hash)
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
