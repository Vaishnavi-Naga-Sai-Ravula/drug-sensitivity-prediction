import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from sklearn.model_selection import train_test_split
from pathdsp.data import build_dataset
from pathdsp.model import PathDSPFNN,standardize_fit,standardize_transform,predict_scaled,regression_metrics
import argparse
parser=argparse.ArgumentParser(description='Train a research MSE-FNN candidate without overwriting the serving model.')
parser.add_argument('--output',type=Path,default=Path('.local_training/mse_candidate'))
args=parser.parse_args()
root=args.output.resolve()
if root==Path(__file__).resolve().parents[1] or (root/'artifacts').exists():raise ValueError('Choose a new candidate directory, not the serving checkout or an existing bundle.')
root.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(2026)
b=build_dataset('GDSC');train,test=train_test_split(np.arange(len(b.y)),test_size=.2,random_state=42);fit,valid=train_test_split(train,test_size=.1,random_state=42)
med,mean,scale=standardize_fit(b.X[fit]);x=standardize_transform(b.X,med,mean,scale)
m=PathDSPFNN(x.shape[1])
for layer in m.modules():
 if isinstance(layer,torch.nn.Linear):torch.nn.init.xavier_uniform_(layer.weight);torch.nn.init.zeros_(layer.bias)
opt=torch.optim.AdamW(m.parameters(),lr=.0004,weight_decay=.0001)
loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x[fit]),torch.from_numpy(b.y[fit,None])),batch_size=256,shuffle=True)
xv=torch.from_numpy(x[valid]);yv=torch.from_numpy(b.y[valid,None]);history=[];best=float('inf');stale=0;start=time.perf_counter()
for epoch in range(60):
 m.train();total=0
 for xb,yb in loader:
  opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(m(xb),yb);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();total+=loss.item()*len(xb)
 m.eval()
 with torch.no_grad():vr=float(torch.nn.functional.mse_loss(m(xv),yv).sqrt())
 history.append({'epoch':epoch+1,'training_rmse':float(np.sqrt(total/len(fit))),'validation_rmse':vr})
 print(history[-1],flush=True)
 if vr<best-1e-5:best=vr;state=copy.deepcopy(m.state_dict());stale=0
 else:stale+=1
 if stale>=10:break
m.load_state_dict(state);m.eval();p=predict_scaled(m,x[test],batch_size=512)
cc=build_dataset('CCLE');ccx=standardize_transform(cc.X,med,mean,scale);cp=predict_scaled(m,ccx,batch_size=512)
row=b.meta[(b.meta.drug=='5-Fluorouracil')&(b.meta.cell=='SIDM00003')].index[0];specific=float(predict_scaled(m,x[row:row+1])[0])
report={'method':'same PathDSP FNN; Xavier initialization; AdamW; MSE objective; two CPU threads; early stopping on inner validation only','fit_rows':len(fit),'validation_rows':len(valid),'test_rows':len(test),'test_split_seed':42,'training_seed':2026,'epochs':len(history),'best_validation_rmse':best,'holdout':regression_metrics(b.y[test],p,float(np.median(b.y[fit]))),'ccle':regression_metrics(cc.y,cp,None),'example':{'drug':'5-Fluorouracil','cell':'SIDM00003','observed':float(b.y[row]),'predicted':specific,'scope':'fit' if row in fit else 'validation' if row in valid else 'test'},'seconds':time.perf_counter()-start,'scope':'Exploratory comparison on a previously used holdout; not independent final certification.'}
from pathdsp.model import save_model_bundle
save_model_bundle(m,b.feature_names,b.feature_groups,med,mean,scale,float(np.median(b.y[fit])),root/'artifacts',training_metrics=report['holdout']);(root/'mse_candidate_results.json').write_text(json.dumps(report,indent=2));pd.DataFrame(history).to_csv(root/'mse_training_history.csv',index=False);print(json.dumps(report,indent=2),flush=True)
