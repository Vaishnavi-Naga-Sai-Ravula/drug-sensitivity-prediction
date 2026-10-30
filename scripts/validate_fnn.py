"""Fit fresh FNN candidates on the frozen benchmark roles, never deploy them."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from pathdsp.data import build_dataset
from pathdsp.model import PathDSPFNN,standardize_fit,standardize_transform,predict_scaled
from pathdsp.research import conformal_radius,ranking_metrics,dataset_fingerprint


def run(directory,epochs=40):
    protocol=json.loads((directory/'protocol.json').read_text())
    bundle=build_dataset('GDSC')
    if dataset_fingerprint(bundle)!=protocol['dataset_sha256']:raise ValueError('Frozen dataset mismatch')
    if (directory/'fnn_summary.csv').exists():raise ValueError('Refusing to overwrite FNN evaluation')
    torch.set_num_threads(2)
    summary=[];all_ranks=[]
    for mode in ('pair','drug','cell','both'):
        membership=pd.read_csv(directory/f'{mode}_membership.csv')
        if not membership[['drug','cell']].equals(bundle.meta[['drug','cell']].reset_index(drop=True)):raise ValueError('Pair identity/order mismatch')
        parts={role:np.flatnonzero(membership.role.eq(role)) for role in ('fit','tune','calibration','test')}
        torch.manual_seed(protocol['seed'])
        med,mean,scale=standardize_fit(bundle.X[parts['fit']]);x=standardize_transform(bundle.X,med,mean,scale)
        model=PathDSPFNN(x.shape[1])
        for layer in model.modules():
            if isinstance(layer,torch.nn.Linear):torch.nn.init.xavier_uniform_(layer.weight);torch.nn.init.zeros_(layer.bias)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.0001)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x[parts['fit']]),torch.from_numpy(bundle.y[parts['fit'],None])),batch_size=256,shuffle=True)
        best=float('inf');stale=0;history=[];start=time.perf_counter()
        for epoch in range(epochs):
            model.train()
            for features,target in loader:
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(features),target)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            p=predict_scaled(model,x[parts['tune']]);score=float(root_mean_squared_error(bundle.y[parts['tune']],p))
            history.append({'epoch':epoch+1,'tune_rmse':score})
            if score<best-1e-5:best=score;state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
            if (epoch+1)%10==0:print(mode,'epoch',epoch+1,'tune RMSE',score,flush=True)
            if stale>=8:break
        model.load_state_dict(state);model.eval()
        p=predict_scaled(model,x[parts['test']]);actual=bundle.y[parts['test']]
        radius=conformal_radius(bundle.y[parts['calibration']],predict_scaled(model,x[parts['calibration']]))
        frame=bundle.meta.iloc[parts['test']].reset_index(drop=True).copy()
        frame['actual_neglog10_ic50']=actual;frame['predicted_neglog10_ic50']=p
        frame['lower_90']=p-radius;frame['upper_90']=p+radius;frame.to_csv(directory/f'{mode}_fresh_fnn_predictions.csv',index=False)
        pd.DataFrame(history).to_csv(directory/f'{mode}_fresh_fnn_history.csv',index=False)
        ranks=ranking_metrics(frame,actual,p);ranks['split']=mode;ranks['model']='fresh_fnn';all_ranks.append(ranks)
        row={'split':mode,'model':'fresh_fnn','fit_rows':len(parts['fit']),'test_rows':len(actual),'tune_rmse':best,
             'mae':float(mean_absolute_error(actual,p)),'rmse':float(root_mean_squared_error(actual,p)),'r2':float(r2_score(actual,p)),
             'interval_radius_90':radius,'coverage_90':float(np.mean(np.abs(actual-p)<=radius)),
             'median_fold_error':float(np.median(10**np.minimum(np.abs(actual-p),20))),
             'mean_cell_top5_overlap':None if ranks.empty else float(ranks.top_k_overlap_fraction.mean()),'seconds':time.perf_counter()-start}
        summary.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame(summary).to_csv(directory/'fnn_summary.csv',index=False)
    pd.concat(all_ranks,ignore_index=True).to_csv(directory/'fnn_within_cell_rankings.csv',index=False)
    import hashlib
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob('*fresh_fnn*.csv'))}
    files.update({name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('fnn_summary.csv','fnn_within_cell_rankings.csv')})
    (directory/'fnn_protocol.json').write_text(json.dumps({'seed':protocol['seed'],'max_epochs':epochs,'patience':8,'optimizer':'AdamW lr=.0004 weight_decay=.0001','initialization':'Xavier','objective':'MSE','fit_preprocessing_only':True,'selection':'early stopping on tuning RMSE only','serving_weights_changed':False,'files':files},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args();run(args.directory,args.epochs)
