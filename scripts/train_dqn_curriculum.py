"""Train fresh DQNs, advancing only when both independent runs retain mastery."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from rl.curriculum import STAGES
from rl.dqn_curriculum import AuditedReplay, BurstExploration, EVENTS, RetainedReplay, make_curriculum_env
from rl.interaction_dqn import PacmanQNetwork, bellman_targets, load_dqn, pack_observation
from rl.pacman_env import PacmanEnv
from scripts.navigation_study import reachable_collectibles
from scripts.train_interaction_dqn import CAPACITY, GAMMA, MAP_SHA256, save_checkpoint, save_json


BUDGET = 131072
INTERVAL = 32768
SEEDS = (211, 907)
GATE = {0: 1, 1: 20, 2: 20, 4: 20, 7: 15}


class Episode:
    def __init__(self, raw):
        self.start = raw
        self.previous = None
        self.stagnation = 0
        self.longest = 0
        self.recent_wall = 0
        self.events = np.zeros(len(EVENTS), dtype=np.int64)
        self.steps = 0
        self.random_actions = 0
        self.first_useful_wall = None
        self.items_after_wall = 0

    def step(self, before, after, info, random_action):
        self.steps += 1
        self.random_actions += random_action
        items = (before["points_remaining"] + before["coins_remaining"]
                 - after["points_remaining"] - after["coins_remaining"])
        lemon = len(before["bonus_positions"][2]) > len(after["bonus_positions"][2])
        wall = len(before["walls"]) > len(after["walls"])
        useful = wall and reachable_collectibles({**before, "walls": after["walls"]}) > reachable_collectibles(before)
        returned = after["pacman"] == self.previous != before["pacman"]
        self.previous = before["pacman"]
        self.stagnation = 0 if items else self.stagnation+1
        self.longest = max(self.longest, self.stagnation)
        self.recent_wall = 30 if useful else max(0, self.recent_wall-1)
        if self.first_useful_wall is not None:
            self.items_after_wall += items
        elif useful:
            self.first_useful_wall = self.steps
        tags = np.array([items > 0, lemon, wall, useful, self.recent_wall > 0, returned,
                         info["reward_terms"]["health"] < 0, info["victory"]], dtype=np.bool_)
        self.events += tags
        return tags

    def result(self, game_seed, info):
        return dict(seed=game_seed, steps=self.steps, random_actions=self.random_actions,
                    score=info["score"], victory=bool(info["victory"]), death=bool(info["game_over"]),
                    timeout=info["termination_reason"] == "timeout",
                    points=self.start["points_remaining"]-info["points_remaining"],
                    coins=self.start["coins_remaining"]-info["coins_remaining"],
                    longest_stagnation=self.longest, first_useful_wall=self.first_useful_wall,
                    items_after_wall=self.items_after_wall,
                    events=dict(zip(EVENTS, self.events.tolist())))


def play(model, ghosts, seed, epsilon=0., render=False, trace=False):
    env = make_curriculum_env(ghosts, "human" if render else None)
    base = env.unwrapped
    rng = np.random.default_rng(np.random.SeedSequence([9876, seed, ghosts]))
    rows = []
    try:
        obs, _ = env.reset(seed=seed)
        episode = Episode(base._last_raw_observation)
        for step in range(1, 3001):
            before = base._last_raw_observation
            packed = pack_observation(obs)
            mask = base.action_masks()
            random_action = epsilon > 0 and rng.random() < epsilon
            q = None
            if trace:
                with torch.inference_mode():
                    q = model(*[torch.as_tensor(value)[None] for value in packed])[0].tolist()
            action = (int(rng.choice(np.flatnonzero(mask))) if random_action
                      else model.greedy_action(packed, mask))
            obs, reward, terminal, truncated, info = env.step(action)
            after = base._last_raw_observation
            tags = episode.step(before, after, info, random_action)
            if trace:
                rows.append(dict(step=step, pos=before["pacman"], next_pos=after["pacman"],
                                 mask=mask.tolist(), q=q, action=action, score=info["score"],
                                 events=[name for name, on in zip(EVENTS, tags) if on], reward=reward))
            if terminal or truncated:
                break
        result = episode.result(seed, info)
        if result["victory"]:
            assert (result["points"], result["coins"], result["score"]) == (81, 11, 136)
            assert result["events"]["lemon"] and result["events"]["wall"]
        if trace:
            result["trace"] = rows
        return result
    finally:
        env.close()


def validate(path, ghosts):
    model = load_dqn(path)
    count = 1 if ghosts == 0 else 30
    start = 85000+100*ghosts
    nominal = [play(model, ghosts, seed) for seed in range(start, start+count)]
    wins = sum(game["victory"] for game in nominal)
    robust = []
    if ghosts == 0 and wins == 1:
        robust = [play(model, 0, seed, .02) for seed in range(start+10, start+20)]
    robust_wins = sum(game["victory"] for game in robust)
    passed = wins >= GATE[ghosts] and (ghosts != 0 or (
        robust_wins >= 8 and nominal[0]["events"]["useful_wall"] > 0))
    return dict(file=path.name, games=nominal, robust=robust, wins=wins, robust_wins=robust_wins, passed=passed,
                mean_score=float(np.mean([game["score"] for game in nominal])),
                games_with_useful_wall=sum(bool(game["events"]["useful_wall"]) for game in nominal))


def update(model, target, replay, retained, optimizer):
    before, after, actions, rewards, masks, terminal = retained.batch(replay)
    with torch.no_grad():
        expected = bellman_targets(target(*after), masks, rewards, terminal, GAMMA)
    current = model(*before).gather(1, actions[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(current, expected)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite DQN loss.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def learn_stage(seed, ghosts, directory, older):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    stage = STAGES.index(ghosts)
    engine_rng, policy_rng, replay_rng, keep_rng = [np.random.default_rng(part)
        for part in np.random.SeedSequence([seed, ghosts]).spawn(4)]
    low = 900000 + seed % 10 * 10000 + stage * 2000
    engine_seeds = iter(engine_rng.permutation(np.arange(low, low+2000)))
    env = make_curriculum_env(ghosts)
    base = env.unwrapped
    model = load_dqn(older[-1] / "dqn_131072.pt") if older else PacmanQNetwork(base.rows, base.cols)
    model.train()
    initial = hashlib.sha256(b"".join(value.detach().numpy().tobytes()
                                       for value in model.state_dict().values())).hexdigest()
    target = PacmanQNetwork(base.rows, base.cols)
    target.load_state_dict(model.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    policy = BurstExploration(policy_rng)
    retained = RetainedReplay([path / "replay_retained.npz" for path in older], replay_rng)
    directory.mkdir(exist_ok=False, parents=True)
    config = dict(ghosts=ghosts, stage=stage+1, seed=seed, budget=BUDGET, checkpoint_interval=INTERVAL,
                  replay_capacity=CAPACITY, initial_replay_size=0, older_replays=[str(p) for p in older],
                  initial_weights=initial, gamma=GAMMA, warmup=4096, batch_size=64, training_frequency=4,
                  target_interval=2048, learning_rate=.0001, burst_length=[3, 7], reward=PacmanEnv.REWARD_CONFIG,
                  map_sha256=MAP_SHA256, engine_seed_range=[low, low+1999])
    save_json(directory / "config.json", config)
    started = perf_counter()
    checkpoints, probes = [], {}
    updates = 0
    wins = 0
    first_win = None
    try:
        game_seed = int(next(engine_seeds))
        obs, _ = env.reset(seed=game_seed)
        assert len(base._last_raw_observation["ghosts"]) == ghosts
        packed = pack_observation(obs)
        mask = base.action_masks()
        replay = AuditedReplay(CAPACITY, packed, replay_rng)
        episode = Episode(base._last_raw_observation)
        losses = []
        random_moves = 0
        episode_actions = []
        tiles = set()
        totals = np.zeros(len(EVENTS), dtype=np.int64)
        quarter_events = np.zeros_like(totals)
        with ((directory / "episodes.jsonl").open("x", encoding="utf-8") as episodes,
              (directory / "progress.jsonl").open("x", encoding="utf-8") as progress):
            for step in range(1, BUDGET+1):
                # Keep coherent exploration in the later part of training.
                epsilon = 1. if step <= 4096 else max(.20, .85 - .65*(step-4096)/94208)
                action, random_move = policy.choose(model, packed, mask, epsilon)
                before = base._last_raw_observation
                obs, reward, terminal, truncated, info = env.step(action)
                if info["window_closed"]:
                    raise RuntimeError("Unexpected window closure during headless training.")
                episode_actions.append(action)
                after = base._last_raw_observation
                next_packed, next_mask = pack_observation(obs), base.action_masks()
                tags = episode.step(before, after, info, random_move)
                totals += tags
                quarter_events += tags
                random_moves += random_move
                tiles.add(base._grid_position(after["pacman"]))
                replay.add(packed, mask, action, reward, next_packed, next_mask, terminal, truncated,
                           events=tags, step=step)
                labels = []
                for position in ((618, 630), (630, 630)):
                    if before["pacman"] == position:
                        labels.append(f"critical_{position[0]}_{position[1]}")
                if episode.stagnation == 80 and tags[5]:
                    labels.append("oscillation")
                if before["effects"][2] > 0 and sum(mask) > 1:
                    labels.append("active_lemon")
                if tags[4] and not tags[3] and sum(mask) > 1:
                    labels.append("after_opening")
                if step == 1:
                    labels.append("start")
                for label in labels:
                    key = (label, before["pacman"]) if label == "oscillation" else (label, (step-1)//INTERVAL)
                    if key not in probes and (label != "oscillation" or
                                              sum(k[0] == label for k in probes) < 12):
                        probes[key] = (packed, mask.copy(), dict(label=label, position=before["pacman"],
                                                               step=step, action=action))
                packed, mask = next_packed, next_mask
                if step > 4096 and step % 4 == 0:
                    losses.append(update(model, target, replay, retained, optimizer))
                    updates += 1
                if step % 2048 == 0:
                    target.load_state_dict(model.state_dict())
                if info["victory"]:
                    wins += 1
                    first_win = step if first_win is None else first_win
                if terminal or truncated:
                    row = dict(decision=step, **episode.result(game_seed, info))
                    episodes.write(json.dumps(row) + "\n")
                    if info["victory"] and first_win == step:
                        save_checkpoint(directory / "first_training_victory.pt", model, step)
                        save_json(directory / "first_training_victory.json", dict(**row, actions=episode_actions))
                    if step < BUDGET:
                        game_seed = int(next(engine_seeds))
                        obs, _ = env.reset(seed=game_seed)
                        packed, mask = pack_observation(obs), base.action_masks()
                        episode = Episode(base._last_raw_observation)
                        episode_actions = []
                        policy.remaining = 0
                if step % 8192 == 0:
                    row = dict(step=step, epsilon=epsilon, random_moves=random_moves, tiles=len(tiles),
                               events=dict(zip(EVENTS, quarter_events.tolist())), replay=replay.snapshot(),
                               retained=retained.snapshot(), loss=float(np.mean(losses)),
                               seconds=perf_counter()-started)
                    progress.write(json.dumps(row) + "\n")
                    progress.flush()
                    episodes.flush()
                    print(f"{seed=} {ghosts=} {step}/{BUDGET} wins={wins} "
                          f"lemons={totals[1]} openings={totals[3]}", flush=True)
                    losses, tiles = [], set()
                    quarter_events.fill(0)
                    random_moves = 0
                if step % INTERVAL == 0:
                    path = directory / f"dqn_{step:06d}.pt"
                    save_checkpoint(path, model, step)
                    checkpoints.append(path)
    finally:
        env.close()

    kept = replay.save(directory / "replay_retained.npz", keep_rng)
    training = dict(config=config, seconds=perf_counter()-started, first_win=first_win,
                    updates=updates, events=dict(zip(EVENTS, totals.tolist())), saved_replay=kept)
    save_json(directory / "training.json", training)
    # All checkpoints score identical stored observations in the Q diagnostic.
    traces = []
    for checkpoint in checkpoints:
        snapshot = load_dqn(checkpoint)
        for observation, legal, details in probes.values():
            with torch.inference_mode():
                values = snapshot(*[torch.as_tensor(value)[None] for value in observation])[0].tolist()
            traces.append(dict(**details, checkpoint=checkpoint.name, mask=legal.tolist(), q=values))
    save_json(directory / "fixed_q_probes.json", traces)
    candidates = checkpoints.copy()
    if first_win is not None:
        candidates.append(directory / "first_training_victory.pt")
    validation = [validate(path, ghosts) for path in candidates]
    selected = max(validation, key=lambda row: (row["wins"], row["robust_wins"],
                                                row["games_with_useful_wall"], row["mean_score"]))
    retention = []
    eligible = all(row["passed"] for row in validation[len(checkpoints)-2:len(checkpoints)])
    if eligible:
        promotion_path = checkpoints[-1]
        retention = [validate(promotion_path, previous) for previous in STAGES[:stage]]
        eligible = all(row["passed"] for row in retention)
    verify_win = next((game for game in selected["games"]+selected["robust"] if game["victory"]), None)
    if verify_win is not None:
        mode = 0. if verify_win in selected["games"] else .02
        replayed = play(load_dqn(directory/selected["file"]), ghosts, verify_win["seed"], mode, trace=True)
        assert replayed["victory"] and replayed["steps"] == verify_win["steps"]
        save_json(directory / "victory_trace.json", replayed)
    result = dict(training=training, validation=validation, retention=retention,
                  selected=selected["file"], passed=eligible)
    save_json(directory / "result.json", result)
    return result


def run(output):
    assert hashlib.sha256(Path("maps/map_2.txt").read_bytes()).hexdigest() == MAP_SHA256
    output.mkdir(exist_ok=False, parents=True)
    save_json(output / "protocol.json", dict(stages=STAGES, budget=BUDGET, seeds=SEEDS,
        wins_required=GATE, games=30, zero_ghost_greedy_games=1, zero_ghost_robust_games=10,
        zero_ghost_robust_wins=8, robust_epsilon=.02, consecutive_checkpoints=2, keep_previous_stages=True))
    older = {seed: [] for seed in SEEDS}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for ghosts in STAGES:
            dirs = [output / f"ghosts_{ghosts}" / f"seed_{seed}" for seed in SEEDS]
            jobs = [pool.submit(learn_stage, seed, ghosts, directory, older[seed])
                    for seed, directory in zip(SEEDS, dirs)]
            results = [job.result() for job in jobs]
            passed = all(result["passed"] for result in results)
            save_json(output / f"gate_{ghosts}.json", dict(ghosts=ghosts, passed=passed,
                      runs=[dict(seed=seed, selected=r["selected"], passed=r["passed"])
                            for seed, r in zip(SEEDS, results)]))
            print(ghosts, "ghosts, gate", passed, flush=True)
            if not passed:
                break
            for seed, directory in zip(SEEDS, dirs):
                older[seed].append(directory)
            if ghosts == 7:
                for directory in dirs:
                    model = load_dqn(directory / "dqn_131072.pt")
                    final = [play(model, 7, seed) for seed in range(86000, 86100)]
                    save_json(directory / "original_game_test.json", final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ghosts", type=int, choices=STAGES, default=0)
    parser.add_argument("--seed", type=int, default=85000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if bool(args.output) == bool(args.model):
        parser.error("Supply exactly one of --output and --model.")
    if args.output:
        run(args.output)
    else:
        torch.set_num_threads(1)
        print(json.dumps(play(load_dqn(args.model), args.ghosts, args.seed, render=args.render), indent=2))


if __name__ == "__main__":
    main()
