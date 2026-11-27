"""Compare saved scenario traces, preserving the raw evidence files."""
import json
from pathlib import Path
import numpy as np
from scripts.navigation_study import reachable_collectibles


ROOT=Path('runs/navigation_study')


def gain_from_observation(path,event):
    with np.load(path) as archive:
        grid=archive['grid'][event['step']-1]
        position=np.rint(archive['pacman_position'][event['step']-1]*[959,719]).astype(int)
    positions=lambda channel:tuple((int(c)*60,int(r)*60) for r,c in np.argwhere(grid[channel]))
    raw=dict(screen_size=(960,720),pacman=tuple(position),walls=positions(0),
             points=positions(2),coins=positions(1))
    removed=set(map(tuple,event['removed_walls']))
    changed={**raw,'walls':tuple(w for w in raw['walls'] if w not in removed)}
    return reachable_collectibles(changed)-reachable_collectibles(raw)


def main():
    rows=[]
    for label,folder in [('original','original_detailed'),('current','current_detailed'),
                         ('mlp_history','mlp_history_detailed'),('spatial_history','spatial_history_detailed')]:
        root=ROOT/folder
        cases=json.loads((root/'scenarios.json').read_text())['cases']
        if label in ('original','current'):
            cases += json.loads((ROOT/'wall_baselines'/f'{label}_summary.json').read_text())
        for case in cases:
            name=case['case']
            if name.startswith(label+'_'):
                path=ROOT/'wall_baselines'/f'{name}.npz'
                name=name[len(label)+1:]
            else:
                path=root/f'{name}_observations.npz'
            events=case.get('wall_events',[])
            gains=[]
            for event in events:
                # Hold starting position and remaining items fixed: this isolates
                # opening topology from concurrent movement and item collection.
                gain=gain_from_observation(path,event)
                assert gain>=0
                gains.append(dict(step=event['step'],removed_walls=event['removed_walls'],
                                  newly_reachable_items=gain,
                                  items_collected_next_50_decisions=event['items_collected_next_50_decisions']))
            rows.append(dict(model=label,case=name,collection=case['collection'],
                             escaped_zone=case['escaped_zone'],returns=case['two_step_returns'],
                             stagnation=case['stagnation_max'],lives_lost=case['lives_lost'],
                             steps=case['steps'],walls_broken=case['walls_broken'],wall_events=gains))
    (ROOT/'scenario_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
