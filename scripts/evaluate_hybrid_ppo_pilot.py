"""Prepare fresh recovery seeds and benchmark the three learning approaches."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rl.imitation_eval import play_network
from rl.tempered_ppo import TemperedCategorical
from scripts.diagnose_dagger_robustness import REFERENCE, verify_references
from scripts.generate_targeted_recoveries import certify_candidate
from scripts.train_hybrid_ppo_pilot import ARMS, FINAL, SEEDS, outcomes
from scripts.train_interaction_dqn import save_json


MODELS = {}


def prepare(path, workers, base_seed=89200, retry_short=False):
    source = json.loads(Path("runs/dagger_targeted_data/holdout.json").read_text())["episodes"]
    families = sorted({r["family"] for r in source})
    jobs = []
    for i, family in enumerate(families):
        rows = [r for r in source if r["family"] == family]
        jobs.extend(dict(family=family, split="test", seed=base_seed+i*20+j, prefix=row["prefix"])
                    for j,row in enumerate(rows[:6]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        attempted = list(pool.map(certify_candidate,jobs))
        if retry_short:
            passed = Counter(row["family"] for row in attempted if row["qualified"])
            retry = [{**job, "seed": job["seed"]+200} for job, row in zip(jobs, attempted)
                     if passed[job["family"]] < 4 and not row["qualified"]]
            attempted.extend(pool.map(certify_candidate, retry))
    selected, counts = [], Counter()
    for row in attempted:
        family = row["family"]
        if row["qualified"] and counts[family] < 4:
            selected.append(row)
            counts[family] += 1
    if not selected or len(counts) < 4:
        raise RuntimeError("Insufficient coverage in held-out recovery.")
    save_json(path,dict(attempted=attempted,selected=selected,counts=counts))
    print("Certified held-out recovery starts",counts,flush=True)


def trial(job):
    name,path,mode,seed,family,prefix,temperature = job
    if path not in MODELS:
        torch.set_num_threads(1)
        MODELS[path] = MaskablePPO.load(path,device="cpu")
    model = MODELS[path]
    model.policy.action_dist = TemperedCategorical(model.action_space.n, temperature or 1.)
    result = play_network(model,seed,danger=False,prefix=prefix,stochastic=bool(temperature))
    if mode == "full" and result["victory"]:
        assert (result["points"],result["coins"],result["score"]) == (81,11,136)
    return dict(model=name, mode=mode, family=family, **result)


def compare(games, control):
    assert [(x["seed"],x["family"]) for x in games] == [(x["seed"],x["family"]) for x in control]
    pairs = np.array([int(a["victory"])-int(b["victory"]) for a,b in zip(games,control)])
    indices = np.random.default_rng(887).integers(len(pairs),size=(10000,len(pairs)))
    return dict(gained=int((pairs>0).sum()),lost=int((pairs<0).sum()),
                delta_points=float(pairs.mean()*100),
                ci95=np.quantile(pairs[indices].mean(axis=1)*100,[.025,.975]).tolist())


def evaluate(root, recovery_path, output, workers):
    verify_references()
    models = {"dagger42":REFERENCE["dagger42"]}
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_{seed}"
            directory = root / name
            result = json.loads((directory / "result.json").read_text())
            assert (result["arm"],result["seed"],result["actual_steps"]) == (arm,seed,result["budget"])
            models[name] = directory / result["selected"]["file"]
            runs[name] = result
    for seed in SEEDS:
        assert len({runs[f"{arm}_{seed}"]["initial_critic"] for arm in ARMS}) == 1
        assert runs[f"hybrid_{seed}"]["initial_actor"] == runs[f"tempered_{seed}"]["initial_actor"]
        assert runs[f"scratch_{seed}"]["initial_actor"] != runs[f"hybrid_{seed}"]["initial_actor"]
    assert len({runs[f"hybrid_{seed}"]["initial_actor"] for seed in SEEDS}) == 1
    recovery = json.loads(recovery_path.read_text())["selected"]
    assert len({row["seed"] for row in recovery}) == len(recovery)
    assert all(row["qualified"] and row["split"]=="test" and row["seed"] not in FINAL for row in recovery)
    output.mkdir(exist_ok=False)
    manifest = dict(full_seeds=FINAL,sampling_seeds=FINAL[:16],
                    models={n:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for n,p in models.items()},
                    recovery=[(row["family"],row["seed"]) for row in recovery])
    save_json(output / "protocol.json",manifest)
    conditions = [("full",seed,"full",[],0.) for seed in FINAL]
    conditions += [("recovery",row["seed"],row["family"],row["prefix"],0.) for row in recovery]
    conditions += [(mode,seed,"full",[],temperature)
                   for mode,temperature in (("sample_t1",1.),("sample_t05",.5)) for seed in FINAL[:16]]
    jobs = [(name,path,*task) for task in conditions for name,path in models.items()]
    results = defaultdict(lambda:defaultdict(list))
    with (ProcessPoolExecutor(max_workers=workers) as pool,
          (output / "games.jsonl").open("x",encoding="utf8") as stream):
        for i,row in enumerate(pool.map(trial,jobs,chunksize=2),1):
            results[row["model"]][row["mode"]].append(row)
            stream.write(json.dumps(row)+"\n")
            if i%70==0:
                stream.flush()
                print(i,"/",len(jobs),flush=True)
    report={}
    for name, modes in results.items():
        report[name] = {mode:dict(**outcomes(rows), paired=compare(rows,results["dagger42"][mode]))
                        for mode,rows in modes.items()}
        print(name,[(mode,r["victory"],r["episodes"]) for mode,r in report[name].items()],flush=True)
    save_json(output / "results.json",report)
    verify_references()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery",type=Path,required=True)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--runs",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    if args.prepare:
        prepare(args.recovery,args.workers)
    elif args.runs is not None and args.output is not None:
        evaluate(args.runs,args.recovery,args.output,args.workers)
    else:
        parser.error("Use --prepare, or specify --runs and --output for the benchmark.")


if __name__ == "__main__":
    main()
