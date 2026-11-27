"""Paired final evaluation and a bounded curriculum checkpoint."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean
import stat

import numpy as np
from sb3_contrib import MaskablePPO

from main.arena_models import ARENA_MODELS, load_model
from rl.ghost_team_env import REWARD_PROFILES, TEAM_SIZES, GhostTeamEnv
from rl.ghost_team_evaluation import play_team, summarize
from scripts.train_ghost_team import BUDGET, REWARD_TEST, SEEDS, TEST, VALIDATION, hash_file, rank
from scripts.train_interaction_dqn import save_json


ACTORS = {}
GHOSTS = {}


def one_game(job):
    opponent, count, name, path, seed, profile = job
    if opponent not in ACTORS:
        ACTORS[opponent] = load_model(ARENA_MODELS[opponent])
    if path is not None and path not in GHOSTS:
        GHOSTS[path] = MaskablePPO.load(path,device="cpu")
    match = GhostTeamEnv(ACTORS[opponent], count, reward_profile=profile)
    try:
        result = play_team(match,seed,GHOSTS[path] if path is not None else name)
    finally:
        match.close()
    assert not result["closed"]
    return dict(opponent=ARENA_MODELS[opponent].name,controller=name,**result)


def differences(a, b, seeds):
    assert [row["seed"] for row in a] == [row["seed"] for row in b] == list(seeds)
    indices = np.random.default_rng(1244).integers(len(seeds),size=(10000,len(seeds)))
    report = {}
    for key in ("ghost_win","lives","controlled_lives","timeout"):
        delta = np.array([float(x[key])-float(y[key]) for x,y in zip(a,b,strict=True)])
        report[key] = dict(delta=mean(delta),ci95=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist())
    for key in ("per_agent", "reversals"):
        delta = np.asarray([row[key] for row in a]) - np.asarray([row[key] for row in b])
        report[key] = dict(delta=delta.mean(axis=0).tolist(),
                           ci95=np.quantile(delta[indices].mean(axis=1), [.025, .975], axis=0).T.tolist())
    return report


def evaluate(directory, count, workers, profile="baseline", reference=None):
    output=directory/"benchmark"
    output.mkdir(exist_ok=False)
    models={"classic":None,"random":None}
    validations={}
    hashes={}
    for seed in SEEDS:
        folder=directory/f"seed_{seed}"
        result=json.loads((folder/"results.json").read_text(encoding="utf-8"))
        assert (result["team_budget"],result["count"],result["seed"])==(BUDGET,count,seed)
        assert result["frozen_verified"] and result["policy_sha256"]==ARENA_MODELS[0].sha256
        assert result["validation_seeds"]==list(VALIDATION) and result["held_out_opponent"]==ARENA_MODELS[1].name
        assert result["reward"] == REWARD_PROFILES[profile]
        name=f"ppo_{seed}"
        validations[name]=result["selected"]
        models[name]=folder/result["selected"]["path"]
        for file in folder.glob("*.zip"):
            file.chmod(stat.S_IREAD)
            hashes[str(file)]=hash_file(file)
        if reference:
            prior_dir = reference / f"seed_{seed}"
            prior = json.loads((prior_dir/"results.json").read_text(encoding="utf-8"))
            assert prior["reward"] == REWARD_PROFILES["baseline"]
            for field in ("count","seed","world_seed","team_budget","agent_budget","held_out_opponent","policy_sha256",
                          "validation_seeds","settings","network","environment","continued_from","initial_sha256"):
                assert prior[field] == result[field], field
            historical = prior_dir / prior["selected"]["path"]
            models[f"reference_{seed}"] = historical
            hashes[str(historical)] = hash_file(historical)
    chosen=max(validations,key=lambda name:rank(validations[name]))
    seeds=tuple(seed+100*(count-2) for seed in (REWARD_TEST if profile == "balanced" else TEST))
    save_json(output/"protocol.json",dict(count=count, test_seeds=seeds, validation_candidate=chosen,
        reward_profile=profile, reference=str(reference) if reference else None,
        checkpoints={name:dict(path=str(path),sha256=hashes[str(path)]) for name,path in models.items() if path},
        protected=hashes, held_out_opponent=ARENA_MODELS[1].name))
    jobs=[(opponent,count,name,path,seed,profile) for seed in seeds for opponent in range(len(ARENA_MODELS))
          for name,path in models.items()]
    rows=defaultdict(lambda:defaultdict(list))
    with (ProcessPoolExecutor(max_workers=workers) as pool,
          (output/"games.jsonl").open("x",encoding="utf-8") as stream):
        for n,row in enumerate(pool.map(one_game,jobs,chunksize=2),1):
            rows[row["opponent"]][row["controller"]].append(row)
            stream.write(json.dumps(row)+"\n")
            if n%40==0:
                print(n,"/",len(jobs),flush=True)
    summary={opp:{name:summarize(games) for name,games in matrix.items()} for opp,matrix in rows.items()}
    contrasts={opp:{name:{base:differences(games,matrix[base],seeds) for base in ("classic","random")}
                    for name,games in matrix.items() if name.startswith("ppo_")} for opp,matrix in rows.items()}
    demo=all(matrix[chosen]["ghost_win"]>0 and min(matrix[chosen]["per_agent"])>0
             and matrix[chosen]["timeout"]<=matrix["random"]["timeout"]+1 for matrix in summary.values())
    # Advancement requires replicated signs and uncertainty evidence on each opponent.
    replicated=all(all(
        effect["random"]["controlled_lives"]["delta"]>0 and effect["random"]["ghost_win"]["delta"]>=-.10
        and matrix[name]["timeout"]<=matrix["random"]["timeout"]+1 and min(matrix[name]["per_agent"])>0
        for name,effect in contrasts[opp].items()) and any(
        effect["random"]["controlled_lives"]["ci95"][0]>0 for effect in contrasts[opp].values())
        for opp,matrix in summary.items())
    reward_effects = {opp: {str(seed): differences(matrix[f"ppo_{seed}"],matrix[f"reference_{seed}"],seeds)
                            for seed in SEEDS} for opp,matrix in rows.items()} if reference else {}
    save_json(output/"results.json",dict(summary=summary,contrasts=contrasts,reward_effects=reward_effects,games=len(jobs),
             demo=demo,advance=replicated,selected=chosen))
    print("Candidate",chosen,"playable",demo,"advance",replicated,flush=True)
    for opponent,variants in summary.items():
        print(opponent, {n:(m["ghost_win"],m["per_agent"],m["timeout"]) for n,m in variants.items()},flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--count",type=int,choices=TEAM_SIZES,required=True)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--reward",choices=REWARD_PROFILES,default="baseline")
    parser.add_argument("--reference",type=Path)
    args=parser.parse_args()
    if args.reference and (args.count != 2 or args.reward != "balanced"):
        parser.error("The reference checkpoint comparison applies to the two-ghost reward ablation.")
    evaluate(args.runs,args.count,args.workers,args.reward,args.reference)


if __name__=="__main__":
    main()
