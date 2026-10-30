"""Reproducible research benchmark; no serving weights or historical outputs overwritten."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from pathdsp.data import build_dataset, CORE_ARCHIVE
from pathdsp.model import standardize_fit, standardize_transform
from pathdsp.research import partition, conformal_radius, ranking_metrics, dataset_fingerprint


def run(output, seed=20261030):
    output.mkdir(parents=True, exist_ok=False)
    bundle = build_dataset('GDSC')
    manifest = {'protocol_version':1, 'seed':seed, 'dataset_sha256':dataset_fingerprint(bundle),
                'archive_sha256':hashlib.sha256(CORE_ARCHIVE.read_bytes()).hexdigest(),
                'scope':'Exploratory research: repository data were previously used for model development. New splits are not an independent final certification.',
                'target':'supplied negLog10IC50; concentration unit not independently verified',
                'selection':'minimum tuning RMSE; test labels never select models; ensemble is fixed equal-weight Ridge/ExtraTrees/HGB',
                'calibration':'90% marginal split-conformal candidate intervals; exchangeability required, not individual or external guarantees',
                'serving_model_changed':False, 'models':{}, 'splits':{}}
    summaries=[];rankings=[]
    with threadpool_limits(limits=2):
        for mode in ('pair','drug','cell','both'):
            parts=partition(bundle.meta,mode,seed)
            membership=bundle.meta.copy();membership['role']='excluded_cross_role'
            for role,indices in parts.items():membership.loc[indices,'role']=role
            membership.to_csv(output/f'{mode}_membership.csv',index=False)
            manifest['splits'][mode]={key:len(value) for key,value in parts.items()}
            med,mean,scale=standardize_fit(bundle.X[parts['fit']])
            x=standardize_transform(bundle.X,med,mean,scale)
            candidates={'mean':DummyRegressor(), 'ridge':Ridge(alpha=100),
                        'extra_trees':ExtraTreesRegressor(n_estimators=64,min_samples_leaf=3,max_features=.6,n_jobs=2,random_state=seed),
                        'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=5,random_state=seed)}
            values={};times={}
            for name,model in candidates.items():
                start=time.perf_counter();model.fit(x[parts['fit']],bundle.y[parts['fit']])
                values[name]={role:model.predict(x[index]) for role,index in parts.items() if role!='fit'}
                times[name]=time.perf_counter()-start
                manifest['models'][name]=model.get_params()
            values['ensemble']={role:np.mean([values[name][role] for name in ('ridge','extra_trees','hist_gradient_boosting')],axis=0) for role in ('tune','calibration','test')}
            times['ensemble']=sum(times[name] for name in ('ridge','extra_trees','hist_gradient_boosting'))
            tuning={name:float(root_mean_squared_error(bundle.y[parts['tune']],value['tune'])) for name,value in values.items()}
            winner=min(tuning,key=tuning.get)
            for name,value in values.items():
                y=bundle.y[parts['test']];p=value['test'];radius=conformal_radius(bundle.y[parts['calibration']],value['calibration'])
                frame=bundle.meta.iloc[parts['test']].copy().reset_index(drop=True)
                frame['actual_neglog10_ic50']=y;frame['predicted_neglog10_ic50']=p
                frame['lower_90']=p-radius;frame['upper_90']=p+radius
                frame.to_csv(output/f'{mode}_{name}_predictions.csv',index=False)
                rank=ranking_metrics(frame,y,p);rank['split']=mode;rank['model']=name;rankings.append(rank)
                row={'split':mode,'model':name,'selected_on_tune':name==winner,'fit_rows':len(parts['fit']),'test_rows':len(y),
                     'tune_rmse':tuning[name],'mae':float(mean_absolute_error(y,p)),'rmse':float(root_mean_squared_error(y,p)),
                     'r2':float(r2_score(y,p)),'interval_radius_90':radius,'coverage_90':float(np.mean(np.abs(y-p)<=radius)),
                     'median_fold_error':float(np.median(10**np.minimum(np.abs(y-p),20))),
                     'mean_cell_top5_overlap':None if rank.empty else float(rank.top_k_overlap_fraction.mean()),'seconds':times[name]}
                summaries.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame(summaries).to_csv(output/'benchmark_summary.csv',index=False)
    pd.concat(rankings,ignore_index=True).to_csv(output/'within_cell_rankings.csv',index=False)
    manifest['files']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob('*.csv'))}
    (output/'protocol.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261030)
    args=parser.parse_args();run(args.output,args.seed)
