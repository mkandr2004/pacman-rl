"""Build transparent comparison artifacts from completed, paired episodes."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
import numpy as np


ROOT=Path('runs/navigation_study')
SOURCES={
    'Ancien dense': 'original_50',
    'Actuel dense': 'current_50',
    'Actuel stochastique': 'current_stochastic_50',
    'Dense + historique': 'mlp_history_50',
    'CNN + historique': 'spatial_history_50',
    'Ancien stochastique': 'original_stochastic_50',
    'Dense historique stochastique': 'mlp_history_stochastic_50',
    'CNN historique stochastique': 'spatial_history_stochastic_50',
}


def main():
    reports={name:json.loads((ROOT/folder/'results.json').read_text())
             for name,folder in SOURCES.items()}
    seeds=[r['seed'] for r in reports['Actuel dense']['results']]
    rows=[]
    for name,report in reports.items():
        results=report['results']
        assert [r['seed'] for r in results]==seeds
        for r in results:
            # Older instrumented batches predate these redundant explicit keys.
            # Display score = points + 5*coins; collection = points + coins.
            r.setdefault('coins_collected',(r['score']-r['collection'])/4)
            r.setdefault('points_collected',r['collection']-r['coins_collected'])
            r.setdefault('survival_sim_seconds',r['simulation_frames']*.01)
            r['stagnation_sim_seconds']=r['stagnation_max']*.04
            r['return_fraction']=r['two_step_returns']/max(1,r['steps']-2)
        fields=('score','collection','points_collected','coins_collected','collection_fraction',
                'two_step_returns','return_fraction','stagnation_sim_seconds','survival_sim_seconds',
                'lives_lost','walls_broken','outside_frames','borders_broken','blocked_frames')
        row=dict(model=name,episodes=len(results),wins=sum(r['victory'] for r in results),
                 deaths=sum(r['game_over'] for r in results),timeouts=sum(r['truncated'] for r in results))
        row.update({key:mean(r[key] for r in results) for key in fields})
        row['score_std']=pstdev(r['score'] for r in results)
        rows.append(row)
    base=np.array([r['score'] for r in reports['Actuel dense']['results']])
    rng=np.random.default_rng(42)
    for row in rows:
        values=np.array([r['score'] for r in reports[row['model']]['results']])-base
        means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
        row['paired_score_difference']=float(values.mean())
        row['paired_score_bootstrap_95']=np.quantile(means,[.025,.975]).tolist()
    (ROOT/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (ROOT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    checkpoints=['runs/resume_20261124_043056/original_model.zip',
                 'runs/ppo_20261124_020053/best_model.zip',
                 'runs/navigation_study/mlp_history/final_model.zip',
                 'runs/navigation_study/spatial_history_v2/final_model.zip']
    hashes={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in checkpoints}
    assert hashes[checkpoints[1]]=='0ec3bf0275fa1bc28a872b098a69fda47d952208cede085762e3f58df1f42c08'
    assert hashes[checkpoints[0]]=='695602e97d6a831c619211aae779b138ee775302351eb2e1ab2d305249538b6f'
    (ROOT/'model_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    labels=['Ancien D','Actuel D','Actuel S','Dense H D','CNN H D','Ancien S','Dense H S','CNN H S']
    for ax,key,title in zip(axes,('score','collection_fraction','return_fraction'),
                           ('Score moyen','Fraction des objets collectés','Retours en deux décisions / décisions')):
        ax.bar(labels,[row[key] for row in rows],color=['#8993a4','#3559a8','#6686c7','#d99729','#299981','#b8bfca','#edbd6a','#6bbca9'])
        ax.tick_params(axis='x',labelrotation=65)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Même carte, sept fantômes, seeds 20000–20049, 50 parties par ligne\nD : déterministe ; S : stochastique ; H : historique')
    fig.savefig(ROOT/'comparison.png',dpi=180)
    plt.close(fig)
    trace=json.loads((ROOT/'current_detailed/zone_160_ghosts_False_stochastic_False.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    x=np.arange(4)
    for index,color in ((0,'#3559a8'),(1,'#d99729')):
        record=trace['decisions'][index]
        axes[0].bar(x+(index-.5)*.34,record['probabilities'],.34,
                    label=str(tuple(record['before'])),color=color)
    axes[0].set(xticks=x,xticklabels=['LEFT','RIGHT','UP','DOWN'],ylim=(0,1),
                title='Probabilités après masque physique')
    axes[0].legend(title='Position avant décision')
    frames=trace['physical_frames'][:24]
    axes[1].plot([0]+[f['frame'] for f in frames],
                 [frames[0]['before'][0]]+[f['after'][0] for f in frames],marker='.')
    axes[1].set(xlabel='Frame physique',ylabel='Coordonnée x (pixels)',
                title='Le mouvement est bien exécuté')
    records=trace['decisions'][:12]
    axes[2].plot([r['step'] for r in records],[-.01]*len(records),label='Récompense de base')
    axes[2].plot([r['step'] for r in records],[r['potential_term'] for r in records],
                 label='Terme PBRS recalculé',marker='.')
    axes[2].axhline(0,color='gray',linewidth=.6)
    axes[2].set(xlabel='Décision',title='Aucune collecte ; potentiel alternant')
    axes[2].legend(fontsize=8)
    fig.suptitle('Boucle reproduite sans fantômes — seed 1000, préfixe commun de 160 décisions')
    fig.savefig(ROOT/'oscillation_mechanism.png',dpi=180)
    plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
