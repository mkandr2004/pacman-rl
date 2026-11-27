"""Train a masked DQN from an empty replay on the original Pac-Man game."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from rl.imitation_env import make_imitation_env
from rl.interaction_dqn import FORMAT, PacmanQNetwork, UniformReplay, bellman_targets, load_dqn, pack_observation
from rl.pacman_env import PacmanEnv


GAMMA = .995
CAPACITY = 65536
CONTROL_STEPS = 2048
TRAIN_STEPS = 131072
CHECKPOINT_INTERVAL = 32768
MAP_SHA256 = "1109e1f8f036055b77f6c386f24e0a6ea0042bfb11e398fac50d192fd6978638"


def save_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def save_checkpoint(path, model, steps):
    with path.open("xb") as stream:
        torch.save(dict(format=FORMAT, model=model.state_dict(), rows=model.rows,
                        cols=model.cols, decisions=steps), stream)


def fingerprint(observation):
    return hashlib.sha256(b"".join(value.tobytes() for value in observation)).hexdigest()


def update(model, target, replay, optimizer):
    states, next_states, actions, rewards, next_masks, terminated = replay.batch(64)
    with torch.no_grad():
        expected = bellman_targets(target(*next_states), next_masks, rewards, terminated, GAMMA)
    current = model(*states).gather(1, actions[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(current, expected)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite DQN loss.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), gradient_norm.item()


def new_metrics():
    return dict(decisions=0, random_actions=0, reward=0., points=0, coins=0, lemons=0, walls=0, lives_lost=0,
                blocked=0, oscillations=0, longest_stagnation=0, wins=0, deaths=0, time_limits=0)


def verify_control(checkpoint, seed, trace):
    model = load_dqn(checkpoint)
    adapter = make_imitation_env(danger=False)
    engine = PacmanEnv(max_episode_steps=3000)
    reloaded_values = None
    try:
        obs, _ = adapter.reset(seed=seed)
        engine.reset(seed=seed)
        for action, expected_obs, expected_mask, expected_reward, terminal, truncated in trace:
            base = adapter.unwrapped
            mask = base.action_masks()
            assert mask[action] and np.array_equal(mask, engine.action_masks())
            assert mask[model.greedy_action(pack_observation(obs), mask)]
            obs, reward, done, cutoff, _ = adapter.step(action)
            _, raw_reward, raw_done, raw_cutoff, _ = engine.step(action)
            assert base._last_raw_observation == engine._last_raw_observation
            assert fingerprint(pack_observation(obs)) == expected_obs
            assert base.action_masks().tolist() == expected_mask
            assert (reward, done, cutoff) == (expected_reward, terminal, truncated)
            assert (reward, done, cutoff) == (raw_reward, raw_done, raw_cutoff)
        reloaded_values = model(*[torch.as_tensor(value)[None] for value in pack_observation(obs)]).detach()
    finally:
        adapter.close()
        engine.close()
    return pack_observation(obs), reloaded_values


def train(seed, output, control):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    engine_rng, action_rng, replay_rng = [np.random.default_rng(child)
                                         for child in np.random.SeedSequence(seed).spawn(3)]
    start_seed = 800000 + (seed % 10) * 10000
    engine_seeds = iter(engine_rng.permutation(np.arange(start_seed, start_seed + 10000)))
    writes = CONTROL_STEPS if control else CHECKPOINT_INTERVAL
    steps = CONTROL_STEPS if control else TRAIN_STEPS
    warmup = 256 if control else 4096
    exploration_end = CONTROL_STEPS if control else 100000
    assert hashlib.sha256(Path("maps/map_2.txt").read_bytes()).hexdigest() == MAP_SHA256
    env = make_imitation_env(danger=False)
    base = env.unwrapped
    started = perf_counter()
    checkpoints, trace = [], []
    updates = 0
    first_win = None
    episode, window = new_metrics(), new_metrics()
    episode_count = 0
    visit_tiles, window_tiles = set(), set()
    totals = new_metrics()
    try:
        game_seed = initial_seed = int(next(engine_seeds))
        current, _ = env.reset(seed=game_seed)
        assert len(base._last_raw_observation["ghosts"]) == 7
        model = PacmanQNetwork(base.rows, base.cols)
        target = PacmanQNetwork(base.rows, base.cols)
        target.load_state_dict(model.state_dict())
        target.requires_grad_(False)
        initial_hash = hashlib.sha256(b"".join(value.detach().numpy().tobytes()
                                                for value in model.state_dict().values())).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        obs = pack_observation(current)
        mask = base.action_masks()
        replay = UniformReplay(CAPACITY, obs, replay_rng)
        config = dict(format=FORMAT, seed=seed, decisions=steps, replay_allocated_bytes=replay.allocated_bytes,
                      replay_capacity=CAPACITY, initial_replay_size=replay.size, initial_model_sha256=initial_hash,
                      gamma=GAMMA, learning_rate=.0001, warmup=warmup, batch_size=64, train_frequency=4,
                      target_interval=2048, exploration_end=exploration_end, final_epsilon=.05,
                      engine_seeds=[start_seed, start_seed + 9999], validation_seeds=list(range(82000, 82030)),
                      map_sha256=MAP_SHA256, original_ghosts=7, reward_config=PacmanEnv.REWARD_CONFIG,
                      python=platform.python_version(), torch=str(torch.__version__), torch_threads=1)
        save_json(output / "config.json", config)
        prev_position = None
        since_pickup = 0
        last_log = started
        batch_losses = []
        batch_gradients = []
        with ((output / "episodes.jsonl").open("x", encoding="utf-8") as episodes,
              (output / "progress.jsonl").open("x", encoding="utf-8") as progress):
            for step in range(1, steps + 1):
                epsilon = 1. - .95 * min(1., max(0., (step - warmup)/(exploration_end - warmup)))
                random_move = action_rng.random() < epsilon
                action = int(action_rng.choice(np.flatnonzero(mask))) if random_move else model.greedy_action(obs, mask)
                before = base._last_raw_observation
                next_current, reward, terminal, truncated, info = env.step(action)
                after = base._last_raw_observation
                next_obs, next_mask = pack_observation(next_current), base.action_masks()
                if info["window_closed"]:
                    raise RuntimeError("The game window closed during headless training.")
                replay.add(obs, mask, action, reward, next_obs, next_mask, terminal, truncated)
                if control and game_seed == initial_seed and step <= 48:
                    trace.append((action, fingerprint(next_obs), next_mask.tolist(), reward, terminal, truncated))
                if step > warmup and step % 4 == 0:
                    loss, gradient = update(model, target, replay, optimizer)
                    batch_losses.append(loss)
                    batch_gradients.append(gradient)
                    updates += 1
                if step % 2048 == 0:
                    target.load_state_dict(model.state_dict())

                points = before["points_remaining"] - after["points_remaining"]
                coins = before["coins_remaining"] - after["coins_remaining"]
                since_pickup = since_pickup + 1 if points + coins == 0 else 0
                x, y = after["pacman"]
                visit_tiles.add((x // 60, y // 60))
                window_tiles.add((x // 60, y // 60))
                events = dict(decisions=1, random_actions=int(random_move), reward=reward, points=points, coins=coins,
                              lemons=len(before["bonus_positions"][2])-len(after["bonus_positions"][2]),
                              walls=len(before["walls"])-len(after["walls"]),
                              lives_lost=round(info["reward_terms"]["health"] / PacmanEnv.REWARD_CONFIG["health"]),
                              blocked=int(before["pacman"] == after["pacman"]),
                              oscillations=int(after["pacman"] == prev_position != before["pacman"]),
                              wins=int(info["victory"]), deaths=int(info["game_over"]),
                              time_limits=int(truncated and not terminal))
                for bucket in (episode, window, totals):
                    for key, value in events.items():
                        bucket[key] += value
                    bucket["longest_stagnation"] = max(bucket["longest_stagnation"], since_pickup)
                prev_position = before["pacman"]
                obs, mask = next_obs, next_mask
                if info["victory"] and first_win is None:
                    first_win = step
                if terminal or truncated:
                    episode_count += 1
                    row = dict(seed=game_seed, decision=step, score=info["score"], health=info["health"],
                               tiles=len(visit_tiles), **episode)
                    episodes.write(json.dumps(row) + "\n")
                    episode, visit_tiles = new_metrics(), set()
                    prev_position, since_pickup = None, 0
                    if step < steps:
                        game_seed = int(next(engine_seeds))
                        obs = pack_observation(env.reset(seed=game_seed)[0])
                        mask = base.action_masks()
                if step % 4096 == 0 or step == steps:
                    now = perf_counter()
                    point = dict(step=step, updates=updates, epsilon=epsilon, tiles=len(window_tiles),
                                 replay_size=replay.size, episodes=episode_count,
                                 seconds=now-started, decisions_per_second=window["decisions"]/(now-last_log),
                                 mean_loss=float(np.mean(batch_losses)) if batch_losses else None,
                                 max_gradient=max(batch_gradients, default=None),
                                 **window)
                    progress.write(json.dumps(point) + "\n")
                    progress.flush()
                    episodes.flush()
                    print(f"seed {seed}  {step}/{steps}  {point['decisions_per_second']:.1f}/s  "
                          f"epsilon {epsilon:.3f}  lemons {window['lemons']}  walls {window['walls']}  "
                          f"wins {totals['wins']}", flush=True)
                    window, window_tiles = new_metrics(), set()
                    batch_losses, batch_gradients = [], []
                    last_log = now
                if step % writes == 0:
                    path = output / f"dqn_{step:06d}.pt"
                    save_checkpoint(path, model, step)
                    checkpoints.append(path)
    finally:
        env.close()

    audit = None
    if control:
        probe, reloaded = verify_control(checkpoints[-1], initial_seed, trace)
        with torch.no_grad():
            assert torch.equal(model(*[torch.as_tensor(value)[None] for value in probe]), reloaded)
        assert updates == (CONTROL_STEPS - warmup)//4 and replay.size == CONTROL_STEPS
        save_json(output / "replay_trace.json", dict(seed=initial_seed, rows=trace))
        final_hash = hashlib.sha256(b"".join(value.detach().numpy().tobytes()
                                            for value in model.state_dict().values())).hexdigest()
        audit = dict(passed=True, replayed=len(trace), parameter_update_verified=final_hash != initial_hash,
            reloaded_q_exact=True, parity_original_engine=True)
        assert audit["parameter_update_verified"]
    save_json(output / "report.json", dict(**config, updates=updates, seconds=perf_counter()-started,
              episodes=episode_count, first_training_win=first_win, totals=totals,
              checkpoints=[p.name for p in checkpoints], audit=audit))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if not args.control:
        if args.preflight is None:
            parser.error("Provide the successful control report with --preflight.")
        proof = json.loads(args.preflight.read_text())
        if proof["decisions"] != CONTROL_STEPS or not proof["audit"]["passed"]:
            parser.error("The DQN control did not pass.")
    args.output.mkdir(parents=True, exist_ok=False)
    train(args.seed, args.output, args.control)


if __name__ == "__main__":
    main()
