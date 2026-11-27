"""Audit a completed DQN curriculum stage, without learning or changing a policy."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.dqn_curriculum import make_curriculum_env
from rl.interaction_dqn import load_dqn
from scripts.train_dqn_curriculum import GAMMA, play
from scripts.train_interaction_dqn import save_json


def replay_victory(path, ghosts, render=False):
    if not path.exists():
        return None
    proof = json.loads(path.read_text())
    env = make_curriculum_env(ghosts, "human" if render else None)
    try:
        env.reset(seed=proof["seed"])
        for index, action in enumerate(proof["actions"], start=1):
            assert env.unwrapped.action_masks()[action]
            _, _, terminated, truncated, info = env.step(action)
            if info["window_closed"]:
                return dict(seed=proof["seed"], decisions=index, certified=False, window_closed=True)
            assert not truncated and terminated == (index == len(proof["actions"]))
        assert info["victory"] and info["points_remaining"] == info["coins_remaining"] == 0
        return dict(seed=proof["seed"], decisions=index, training_step=proof["decision"],
                    score=info["score"], certified=True)
    finally:
        env.close()


def discount(rewards):
    total = 0.
    for reward in reversed(rewards):
        total = reward + GAMMA*total
    return total


def branch(model, ghosts, seed, trace, start, alternative):
    env = make_curriculum_env(ghosts)
    base = env.unwrapped
    rewards = []
    items = score = 0
    try:
        obs, _ = env.reset(seed=seed)
        for event in trace[:start]:
            obs, _, done, limited, info = env.step(event["action"])
            assert not (done or limited)
        before = base._last_raw_observation
        assert list(before["pacman"]) == list(trace[start]["pos"]) and base.action_masks()[alternative]
        initial_items = before["points_remaining"] + before["coins_remaining"]
        initial_score = before["score"]
        action = alternative
        for _ in range(start, 3000):
            obs, reward, done, limited, info = env.step(action)
            rewards.append(reward)
            items = initial_items-info["points_remaining"]-info["coins_remaining"]
            score = info["score"]-initial_score
            if done or limited:
                break
            action, _ = model.predict(obs, action_masks=base.action_masks())
        return dict(action=alternative, discounted_return=discount(rewards), horizon=len(rewards),
                    items=items, score_gain=score, victory=info["victory"])
    finally:
        env.close()


def audit(job):
    directory, ghosts, seed = job
    torch.set_num_threads(1)
    result = json.loads((directory / "result.json").read_text())
    path = directory / result["selected"]
    model = load_dqn(path)
    choice = next(row for row in result["validation"] if row["file"] == path.name)
    game_seed = min(choice["games"], key=lambda g: (
        g["victory"], -g["longest_stagnation"], -g["events"]["return"]))["seed"]
    game = play(model, ghosts, game_seed, trace=True)
    trace = game["trace"]
    score_steps = [0] + [i for i in range(1, len(trace)) if trace[i-1]["score"] != trace[i]["score"]]
    start, end = max(zip(score_steps, score_steps[1:]+[len(trace)]), key=lambda pair: pair[1]-pair[0])
    first_loop = next((i for i in range(start+min(30, end-start-1), end)
                       if "return" in trace[i]["events"] and sum(trace[i]["mask"]) >= 2), None)
    probes = []
    if first_loop is not None:
        event = trace[first_loop]
        trials = [branch(model, ghosts, game_seed, trace, first_loop, action)
                  for action, valid in enumerate(event["mask"]) if valid]
        probes.append(dict(state=event, trials=trials))
    original = [play(model, 7, s) for s in range(87000, 87020)]
    exploration = [play(model, ghosts, s, .20) for s in range(85400, 85408)]
    repetitions = Counter(tuple(row["pos"]) for row in trace[start:end])
    critical = [row for row in trace if tuple(row["pos"]) in ((618, 630), (630, 630))]
    evidence = dict(training_seed=seed, selected=path.name, trajectory_seed=game_seed,
                    stagnation=[start+1, end], repeated_positions=repetitions.most_common(5),
                    q_forks=probes, critical_count=len(critical), autonomous_trace=game,
                    original_game_diagnostic=original, exploration_diagnostic=exploration,
                    first_training_victory=replay_victory(directory / "first_training_victory.json", ghosts))
    return evidence


def draw(runs, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.2), layout="constrained")
    for seed, run in runs:
        x = [row["step"]/1000 for row in run]
        axes[0, 0].plot(x, [row["replay"]["resident"]["lemon"] for row in run], label=str(seed))
        axes[0, 1].plot(x, [row["replay"]["resident"]["useful_wall"] for row in run])
        axes[1, 0].plot(x, [100*row["events"]["return"]/8192 for row in run])
        axes[1, 1].plot(x, [100*row["random_moves"]/8192 for row in run])
    for ax, title in zip(axes.flat, (
        "Transitions citron présentes dans le rejeu", "Ouvertures utiles présentes dans le rejeu",
        "Décisions A–B–A (%)", "Décisions d’exploration (%)",
    )):
        ax.set_title(title)
        ax.set_xlabel("Décisions d’entraînement (milliers)")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(title="Seed", frameon=False)
    figure.savefig(output, dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--campaign", type=Path)
    mode.add_argument("--replay", type=Path)
    parser.add_argument("--ghosts", type=int, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.replay:
        proof = replay_victory(args.replay, args.ghosts, args.render)
        if proof is None:
            parser.error("The victory trace does not exist.")
        print(json.dumps(proof, indent=2))
        return
    gate = json.loads((args.campaign / f"gate_{args.ghosts}.json").read_text())
    jobs, curves = [], []
    for run in gate["runs"]:
        seed = run["seed"]
        directory = args.campaign / f"ghosts_{args.ghosts}" / f"seed_{seed}"
        jobs.append((directory, args.ghosts, seed))
        progress = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
        curves.append((seed, progress))
    output = args.campaign / f"analysis_{args.ghosts}"
    output.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as pool:
        evaluations = list(pool.map(audit, jobs))
    for row in evaluations:
        print(row["training_seed"], row["selected"], row["stagnation"],
              "original wins", sum(g["victory"] for g in row["original_game_diagnostic"]), flush=True)
    save_json(output / "evidence.json", evaluations)
    draw(curves, output / "curves.png")


if __name__ == "__main__":
    main()
