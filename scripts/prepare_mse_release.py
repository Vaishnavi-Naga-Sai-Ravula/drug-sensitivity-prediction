"""Build a consistent research release in a staging directory; does not publish it."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pathdsp.data import build_dataset
from pathdsp.model import load_model_bundle,standardize_transform,predict_scaled,regression_metrics,artifact_digest
from pathdsp.visualizations import save_evaluation_figures,save_target_distribution,save_training_history_figure,save_group_shap_figure
from train import save_group_shap,save_feature_shap_outputs

parser=argparse.ArgumentParser()
parser.add_argument('--candidate',type=Path,required=True,help='Directory containing artifacts/')
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--history',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True,help='New staging directory; never the serving checkout')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
if args.output.resolve()==root or args.output.exists():raise ValueError('Choose a new staging directory.')
args.output.mkdir(parents=True);out=args.output/'outputs';out.mkdir();figures=out/'figures';torch.set_num_threads(2)
b=build_dataset('GDSC');m,meta=load_model_bundle(args.candidate/'artifacts',b.feature_names)
report=json.loads(args.results.read_text());train,test=train_test_split(np.arange(len(b.y)),test_size=.2,random_state=42);fit,valid=train_test_split(train,test_size=.1,random_state=42)
x=standardize_transform(b.X,np.array(meta['imputation_medians']),np.array(meta['scaler_means']),np.array(meta['scaler_scales']))
threshold=float(np.median(b.y[fit]));pred=predict_scaled(m,x[test],batch_size=512);metrics=regression_metrics(b.y[test],pred,threshold)
if abs(metrics['rmse_neglog10_ic50']-report['holdout']['rmse_neglog10_ic50'])>1e-5:raise ValueError('Candidate does not reproduce the training report.')
def prediction_table(bundle,indices,prediction):
 frame=bundle.meta.iloc[indices].reset_index(drop=True).copy();actual=bundle.y[indices]
 frame['actual_neglog10_ic50']=actual;frame['predicted_neglog10_ic50']=prediction;frame['absolute_error']=abs(actual-prediction)
 frame['actual_ic50_uM']=10.**-actual.astype(float);frame['predicted_ic50_uM']=10.**-prediction.astype(float)
 return frame
holdout=prediction_table(b,test,pred);holdout.to_csv(out/'holdout_predictions.csv',index=False)
cc=build_dataset('CCLE');ccx=standardize_transform(cc.X,np.array(meta['imputation_medians']),np.array(meta['scaler_means']),np.array(meta['scaler_scales']));cp=predict_scaled(m,ccx,batch_size=512)
prediction_table(cc,np.arange(len(cc.y)),cp).to_csv(out/'ccle_external_predictions.csv',index=False)
history=pd.read_csv(args.history).rename(columns={'training_rmse':'train_rmse','validation_rmse':'valid_rmse'});history.to_csv(out/'training_history.csv',index=False)
save_evaluation_figures(figures,holdout,metrics,X_test=b.X[test],feature_names=b.feature_names);save_target_distribution(figures,b.y);save_training_history_figure(figures,history)
groups=save_group_shap(m,x[test],b.feature_groups,out);save_group_shap_figure(figures,groups)
rng=np.random.default_rng(49);background=rng.choice(fit,size=16,replace=False);samples=rng.choice(test,size=12,replace=False)
features=save_feature_shap_outputs(m,x[background],x[samples],b.meta.iloc[samples].reset_index(drop=True),b.feature_names,out,nsamples=32)
result={'dataset':'GDSC','examples':len(b.y),'drugs_with_all_drug_features':len(b.drug_ids),'cell_lines_with_all_cell_features':len(b.cell_ids),'feature_count':len(b.feature_names),'feature_groups':{k:len(v) for k,v in b.feature_groups.items()},'split_mode':'pair','test_size':.2,'seed':42,'training_seed':2026,'max_epochs':60,'epochs_used':len(history),'training_recipe':report['method'],'model_scope':'Same evaluated checkpoint is served; no refit on held-out rows. Model selection reused a historical holdout, so this is not independent final certification.','holdout':metrics,'ccle_external':regression_metrics(cc.y,cp,None),'ccle_external_examples':len(cc.y),'feature_shap_examples':12,'feature_shap_background_examples':16,'feature_shap_model':'served checkpoint; group and feature summaries use held-out examples','top_feature_shap':features.head(10).to_dict(orient='records')}
(out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
roles=b.meta.copy();roles['model_split']='fit';roles.loc[valid,'model_split']='validation';roles.loc[test,'model_split']='holdout';roles.to_csv(out/'model_split_membership.csv',index=False)
shutil.copytree(args.candidate/'artifacts',args.output/'artifacts')
meta_path=args.output/'artifacts/metadata.json';newmeta=json.loads(meta_path.read_text());newmeta['training_recipe']=report['method'];newmeta['training_scope']=result['model_scope'];newmeta['training_seed']=2026;newmeta['split_seed']=42;meta_path.write_text(json.dumps(newmeta,indent=2)+'\n')
manifest={name:artifact_digest(args.output/'artifacts'/name) for name in ('model.pt','metadata.json')};(args.output/'artifacts/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
report['release_model_sha256']=manifest['model.pt'];(out/'mse_training_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('Staged matching model, predictions, SHAP, figures and metrics:',args.output)
