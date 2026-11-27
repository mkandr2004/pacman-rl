"""Train paired occupancy predictors on identical engine replays."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl.curriculum import CurriculumEnv
from rl.ghost_forecast import danger_map
from rl.planning_reference import PlanningReference


def occupancy(env):
    grid = np.zeros((env.rows,env.cols),dtype=np.float32)
    ghosts = env._game._screen_field_mapper.get_ghosts()
    for ghost in ghosts:
        rect = ghost.get_rect()
        grid[max(0,rect.top//60):(rect.bottom-1)//60+1,
             max(0,rect.left//60):(rect.right-1)//60+1] = 1
    return grid


def replay(seed):
    env = CurriculumEnv(7,max_episode_steps=500)
    policy = PlanningReference(seed,anticipate=False)
    inputs, outputs, footprints = [], [], []
    try:
        obs,_ = env.reset(seed=seed)
        for step in range(500):
            occupied = occupancy(env)
            footprints.append(occupied)
            if step % 3 == 0:
                velocity = occupied-footprints[-2] if step else np.zeros_like(occupied)
                forecast = danger_map(env._game,env.rows,env.cols)
                inputs.append(np.concatenate((obs["grid"].astype(np.float32),occupied[None],
                                              velocity[None],forecast)))
            action = policy.predict(env)
            obs,_,terminal,truncated,_ = env.step(action)
            if terminal or truncated:
                break
        for step in range(0,len(footprints)-3,3):
            outputs.append(np.maximum.reduce(footprints[step+1:step+4]))
        return np.stack(inputs[:len(outputs)]),np.stack(outputs)
    finally:
        env.close()


def average_precision(probabilities, labels):
    order = np.argsort(-probabilities.ravel(),kind="stable")
    truth = labels.ravel()[order]
    cumulative = np.cumsum(truth)
    # Evaluate precision only after all cells tied at a threshold.
    ordered = probabilities.ravel()[order]
    last = np.flatnonzero(np.r_[ordered[1:]!=ordered[:-1],True])
    return float(np.sum(np.diff(np.r_[0,cumulative[last]])*cumulative[last]/(last+1))/truth.sum())


def fit(train, validation, seed, use_forecast):
    torch.manual_seed(seed)
    x,y = (torch.tensor(part) for part in train)
    test_x,test_y = (torch.tensor(part) for part in validation)
    if not use_forecast:
        x[:,-1] = 0
        test_x[:,-1] = 0
    net = nn.Sequential(nn.Conv2d(x.shape[1],16,3,padding=1),nn.ReLU(),
                        nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0))
    optimizer = torch.optim.Adam(net.parameters(),lr=.003)
    for _ in range(25):
        for indices in torch.randperm(len(x)).split(64):
            optimizer.zero_grad()
            error = loss(net(x[indices]).squeeze(1),y[indices])
            error.backward()
            optimizer.step()
    with torch.no_grad():
        logits = net(test_x).squeeze(1)
        probabilities = logits.sigmoid().numpy()
        return dict(ap=average_precision(probabilities,test_y.numpy()),
                    brier=float(np.mean((probabilities-test_y.numpy())**2)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    train_seeds=list(range(41000,41012))
    test_seeds=list(range(42000,42006))
    train_games=[replay(seed) for seed in train_seeds]
    test_games=[replay(seed) for seed in test_seeds]
    train=tuple(np.concatenate(parts) for parts in zip(*train_games))
    validation=tuple(np.concatenate(parts) for parts in zip(*test_games))
    rows=[]
    for seed in (42,137,241):
        rows.append(dict(seed=seed,without_map=fit(train,validation,seed,False),
                         with_map=fit(train,validation,seed,True)))
    result=dict(task="predict ghost tile occupancy in the next three decisions",
                train_seeds=train_seeds,test_seeds=test_seeds,train_examples=len(train[0]),
                test_examples=len(validation[0]),predictions=rows,
                evaluation_scope="supervised prediction, not policy or victory")
    (args.output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
