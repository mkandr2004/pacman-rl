"""Compare partial and full DAgger updates using validation seeds only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import torch
from sb3_contrib import MaskablePPO

from scripts.train_dagger import prefixes_from_manifest, ranking, validate, verify_reference


FRACTIONS = (.05, .15, .35, .65)


def score_fraction(job):
    directory, checkpoint, fraction, prefixes = job
    torch.set_num_threads(1)
    model = MaskablePPO.load(directory / checkpoint, device="cpu")
    return dict(fraction=fraction, checkpoint=checkpoint, validation=validate(model, prefixes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_reference()
    report = json.loads((args.run / "training_report.json").read_text())
    curve = json.loads((args.run / "curve.json").read_text())
    chosen_round = next(row for row in curve if row["checkpoint"] == report["selected"])
    manifest = json.loads(Path("runs/expert_imitation_120/manifest.json").read_text())
    prefixes = prefixes_from_manifest(manifest)
    initial = MaskablePPO.load(report["initial"], device="cpu")
    updated = MaskablePPO.load(args.run / report["selected"], device="cpu")
    before = initial.policy.state_dict()
    after = {key: value.clone() for key, value in updated.policy.state_dict().items()}
    assert before.keys() == after.keys()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.run / "config.json", args.output / "config.json")
    shutil.copyfile(args.run / "curve.json", args.output / "curve.json")
    shutil.copyfile(args.run / report["selected"], args.output / "full_update.zip")
    assessments = [dict(fraction=1., checkpoint="full_update.zip",
                        validation=chosen_round["validation"])]
    jobs = []
    for fraction in FRACTIONS:
        # The parent is the exact BC initialization used for this trajectory
        # series. Frozen critic weights and non-floating buffers stay intact.
        adjusted = {
            key: before[key].lerp(value, fraction) if value.is_floating_point()
            and not torch.equal(before[key], value) else value
            for key, value in after.items()
        }
        updated.policy.load_state_dict(adjusted)
        checkpoint = f"fraction_{int(100*fraction):02}.zip"
        updated.save(args.output / checkpoint)
        jobs.append((args.output, checkpoint, fraction, prefixes))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for assessment in pool.map(score_fraction, jobs):
            assessments.append(assessment)
            result = assessment["validation"]
            print(assessment["fraction"], result["full"]["victory"], result["recovery_wins"], flush=True)
    selected = max(assessments, key=lambda row: ranking(row["validation"]))
    (args.output / "interpolation.json").write_text(json.dumps(assessments, indent=2))
    (args.output / "training_report.json").write_text(json.dumps(
        {**report, "parent_run": str(args.run), "unscaled_checkpoint": report["selected"],
         "selected": selected["checkpoint"], "fraction": selected["fraction"]}, indent=2))
    verify_reference()


if __name__ == "__main__":
    main()
