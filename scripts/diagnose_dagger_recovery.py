"""Compare autonomous trajectories and verified expert continuations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from rl.imitation_env import make_imitation_env
from scripts.diagnose_dagger_robustness import (
    REFERENCE, describe_state, model_for, policy_logits, run_episode, verify_references,
)
from scripts.navigation_study import reachable_collectibles
from scripts.ppo_recovery_states import certify


def certificates():
    names = ("ppo_recovery_verified", "ppo_recovery_extra")
    return [r for n in names for r in json.loads((Path("runs") / n / "qualified.json").read_text())
            if r["qualified"] and r["split"] == "test"]


def compare(job):
    case, name = job
    seed = case["seed"]
    prefix = case["prefix"]
    expert_actions = case["winning_actions"]
    autonomous, player, _ = run_episode(name, seed, prefix=prefix)
    expert, teacher, _ = run_episode(name, seed, prefix=prefix, force=expert_actions)
    assert expert["victory"] and len(teacher) == len(expert_actions)
    divergences = np.flatnonzero(player[:min(len(player), len(teacher)), 4]
                                 != teacher[:min(len(player), len(teacher)), 4])
    if not len(divergences):
        return dict(model=name, seed=seed, autonomous=autonomous, expert=expert)
    divergence = int(divergences[0])
    env = make_imitation_env(danger=False, prefix=prefix + expert_actions[:divergence])
    try:
        obs, _ = env.reset(seed=seed)
        base = env.unwrapped
        snapshot = describe_state(base)
        snapshot.update(reachable=reachable_collectibles(base._last_raw_observation),
                        action_mask=base.action_masks().tolist(),
                        observation={key: np.asarray(value).tolist() for key, value in obs.items()})
        assert snapshot["position"] == tuple(player[divergence, 6:8])
        assert np.allclose(policy_logits(model_for(name), obs, base.action_masks()).softmax(0).numpy(),
                           player[divergence, :4])
    finally:
        env.close()
    loss_steps = (np.flatnonzero(player[:, 10] < player[:, 8]) + 1).tolist()
    teacher_loss_steps = (np.flatnonzero(teacher[:, 10] < teacher[:, 8]) + 1).tolist()
    checkpoints = sorted(set([divergence + 1, *[s - 1 for s in loss_steps if s > 0]]))
    rescue = []
    for stop in checkpoints:
        qualified = certify(("audit", "diagnostic", seed, prefix + player[:stop, 4].astype(int).tolist()))
        rescue.append(dict(at=stop, qualified=qualified["qualified"],
                           facts={k:v for k,v in qualified.items() if k not in ("prefix", "winning_actions")}))
    return dict(model=name, seed=seed, first_divergence=divergence+1,
                snapshot=snapshot, probabilities=player[divergence, :4].tolist(),
                neural_action=int(player[divergence, 4]), expert_action=int(teacher[divergence, 4]),
                neural_life_losses=loss_steps, expert_life_losses=teacher_loss_steps,
                autonomous=autonomous, expert=expert, rescue=rescue,
                trace_neural=player.tolist(), trace_expert=teacher.tolist())


def calibration_rows(case):
    env = make_imitation_env(danger=False, prefix=case["prefix"])
    scores = {name: [] for name in REFERENCE}
    try:
        obs, _ = env.reset(seed=case["seed"])
        for index, action in enumerate(case["winning_actions"]):
            mask = env.unwrapped.action_masks()
            assert mask[action]
            if index % 6 == 0 and sum(mask) > 1:
                for name in REFERENCE:
                    prob = policy_logits(model_for(name), obs, mask).softmax(0).tolist()
                    scores[name].append([*prob, action])
            obs, _, terminal, truncated, info = env.step(action)
            assert not truncated and terminal == (index + 1 == len(case["winning_actions"]))
        assert info["victory"] and info["coins_remaining"] == info["points_remaining"] == 0
    finally:
        env.close()
    return dict(seed=case["seed"], family=case["family"], scores=scores)


def calibration_summary(rows):
    report = {}
    for name in REFERENCE:
        data = np.array([row for case in rows for row in case["scores"][name]], dtype=np.float64)
        batch = np.arange(len(data))
        labels = data[:, 4].astype(int)
        result = []
        # These probabilities predict agreement with one winning expert action.
        for temperature in (.25, .5, .75, 1., 1.25):
            p = data[:, :4] ** (1 / temperature)
            p /= p.sum(axis=1, keepdims=True)
            confidence = p.max(axis=1)
            correct = p.argmax(axis=1) == labels
            buckets = []
            for k in range(10):
                belongs = np.minimum((confidence * 10).astype(int), 9) == k
                if belongs.any():
                    count = int(belongs.sum())
                    buckets.append(dict(interval=k, count=count, confidence=float(confidence[belongs].mean()),
                                        accuracy=float(correct[belongs].mean())))
            result.append(dict(temperature=temperature, states=len(data), accuracy=float(correct.mean()),
                               ece=sum(x["count"] * abs(x["confidence"] - x["accuracy"])
                                       for x in buckets) / len(data),
                               nll=float(-np.log(np.maximum(p[batch, labels], 1e-30)).mean()),
                               brier=float(((p - np.eye(4)[labels]) ** 2).sum(axis=1).mean()),
                               buckets=buckets))
        report[name] = result
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    verify_references()
    args.output.mkdir(parents=True, exist_ok=False)
    archived = certificates()
    focus = [case for case in archived if case["family"] == "prefix_161"]
    assert len(focus) == 6 and all(case["position"] == [618, 630] for case in focus)
    jobs = [(case, name) for case in focus for name in REFERENCE]
    audit_cases = []
    for family in sorted({row["family"] for row in archived}):
        audit_cases += [row for row in archived if row["family"] == family][:3]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        samples = list(pool.map(calibration_rows, audit_cases))
        (args.output / "calibration_samples.json").write_text(json.dumps(samples, indent=2))
        (args.output / "calibration.json").write_text(json.dumps(calibration_summary(samples), indent=2))
        with (args.output / "critical.jsonl").open("x") as stream:
            for row in pool.map(compare, jobs):
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(row["model"], row["seed"], row.get("first_divergence"), flush=True)


if __name__ == "__main__":
    main()
