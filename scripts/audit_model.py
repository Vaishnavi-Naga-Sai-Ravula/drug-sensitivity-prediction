import sys,json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np,torch
from pathdsp.data import build_dataset
from pathdsp.model import load_model_bundle, standardize_transform,predict_scaled,regression_metrics
root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
report={}
for source in ('GDSC','CCLE'):
 b=build_dataset(source);m,meta=load_model_bundle(root/'artifacts',b.feature_names)
 x=standardize_transform(b.X,np.array(meta['imputation_medians']),np.array(meta['scaler_means']),np.array(meta['scaler_scales']))
 p=predict_scaled(m,x,batch_size=512)
 report[source]={'scope':('saved deployment model; GDSC includes development/training rows, not independent validation' if source == 'GDSC' else 'saved GDSC-trained deployment model evaluated externally on CCLE; no CCLE fitting'),'metrics':regression_metrics(b.y,p,None),'median_absolute_log_error':float(np.median(abs(b.y-p))),'median_concentration_fold_error':float(10**np.median(abs(b.y-p)))}
print(json.dumps(report,indent=2))
(root/'outputs'/'deployment_model_audit.json').write_text(json.dumps(report,indent=2)+'\n')
