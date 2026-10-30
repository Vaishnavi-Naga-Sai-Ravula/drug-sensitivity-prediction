"""Cell-line research comparisons and self-contained, auditable evaluation exports."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from html import escape
from io import BytesIO
from pathlib import Path
import base64
import hashlib
import json
import re
import zipfile
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .explain import exact_group_shap, shap_feature_values
from .inference import infer_features, pair_scope
from .ic50 import interpretation_data, INTERPRETATION, PLOT_LOCK

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
LIMITATION = ('Research cell-line comparison only. Lower predicted IC50 does not establish a safer or more effective treatment. '
              'Patient health status, toxicity, achievable exposure and patient outcomes are unavailable. '
              'CCLE transfer is poor (deployment-model external R² < 0). SHAP describes model associations, not causation.')


def sample_features(bundle, count=5):
    """Real feature rows, no response target; directly accepted by the upload contract."""
    pairs = bundle.meta.head(count)
    values = np.stack([bundle.pair_features(row.drug, row.cell) for row in pairs.itertuples()])
    frame = pd.DataFrame(values, columns=bundle.feature_names)
    frame.insert(0, 'cell', pairs['cell'].to_numpy())
    frame.insert(0, 'drug', pairs['drug'].to_numpy())
    frame.insert(0, 'dataset', bundle.name)
    return frame


def build_evaluation(bundle, model, metadata, drug, cell):
    raw = np.stack([bundle.pair_features(candidate, cell) for candidate in bundle.drug_ids])
    predictions, scaled, shifted = infer_features(raw, model, metadata)
    observed = [bundle.observed_response(candidate, cell) for candidate in bundle.drug_ids]
    ranks = pd.DataFrame({'drug':bundle.drug_ids, 'cell':cell, 'dataset':bundle.name,
                          'predicted_neglog10_ic50':predictions, 'observed_neglog10_ic50':observed})
    ranks['observed_neglog10_ic50'] = pd.to_numeric(ranks['observed_neglog10_ic50'], errors='coerce').astype(float)
    ranks['predicted_ic50_uM_assuming_source_unit'] = 10.0 ** -np.clip(predictions, -20, 20)
    ranks['observed_ic50_uM_assuming_source_unit'] = 10.0 ** -np.clip(ranks['observed_neglog10_ic50'], -20, 20)
    ranks['absolute_log_error'] = (ranks['predicted_neglog10_ic50'] - ranks['observed_neglog10_ic50']).abs()
    ranks['concentration_fold_error'] = 10.0 ** ranks['absolute_log_error'].clip(upper=20)
    ranks['patient_health_status'] = 'Unavailable — cell-line dataset; no patient records'
    ranks['evidence_scope'] = [bundle.name+' · '+pair_scope(bundle.name,candidate,cell) for candidate in bundle.drug_ids]
    ranks['selected_drug'] = ranks['drug'].eq(drug)
    ranks = ranks.sort_values(['predicted_ic50_uM_assuming_source_unit','drug'], kind='stable').reset_index(drop=True)
    ranks.insert(0, 'research_rank', np.arange(1, len(ranks)+1))
    selected = ranks.loc[ranks['selected_drug']].copy()
    if len(selected) != 1:
        raise ValueError('Selected drug is not present in this dataset.')
    chosen = bundle.drug_index[drug.strip().casefold()]
    x = scaled[chosen]
    groups = exact_group_shap(model, x, metadata['feature_groups'])
    group_frame = pd.DataFrame({'feature_group':list(groups), 'shap_value':list(groups.values())})
    group_frame['absolute_shap'] = group_frame['shap_value'].abs()
    group_frame = group_frame.sort_values('absolute_shap', ascending=False)
    # Standardized zero is the exact attribution baseline (raw training means may contain no missing values).
    from .model import predict_scaled
    baseline = float(predict_scaled(model, np.zeros_like(x).reshape(1,-1))[0])
    feature_status = 'available: approximate Gradient SHAP, 32 integration samples, 16 GDSC background pairs'
    try:
        from .inference import dataset_resource
        reference = dataset_resource('GDSC')
        rng = np.random.default_rng(17)
        indices = rng.choice(len(reference.meta), size=min(16,len(reference.meta)), replace=False)
        background = np.stack([reference.pair_features(reference.meta.iloc[i]['drug'],reference.meta.iloc[i]['cell']) for i in indices])
        from .model import standardize_transform
        background = standardize_transform(background,np.asarray(metadata['imputation_medians']),np.asarray(metadata['scaler_means']),np.asarray(metadata['scaler_scales']))
        values = shap_feature_values(model, background, x, nsamples=32)
        features = pd.DataFrame({'feature':metadata['feature_names'], 'shap_value':values})
        features['feature_group'] = features['feature'].str.split('__').str[0]
        features['absolute_shap'] = features['shap_value'].abs()
        features = features.sort_values('absolute_shap', ascending=False)
    except (ImportError, RuntimeError) as exc:
        feature_status = f'Feature SHAP unavailable: {exc}. Exact group SHAP remains available.'
        features = pd.DataFrame(columns=['feature','shap_value','feature_group','absolute_shap'])
    pathways = features[features['feature_group'] != 'CHEM'].copy()
    tables = {'selected_pair':selected, 'cell_line_drug_rankings':ranks, 'local_group_shap':group_frame,
              'local_feature_shap':features, 'local_pathway_shap':pathways}
    for path in sorted(OUTPUTS.glob('*.csv')):
        tables['saved_' + path.stem] = pd.read_csv(path)
    from .interactive import add_chart_source_tables
    metrics = json.loads((OUTPUTS/'metrics.json').read_text(encoding='utf-8'))
    audits = {path.stem:json.loads(path.read_text(encoding='utf-8')) for path in sorted(OUTPUTS.glob('*.json')) if path.name != 'metrics.json'}
    context = {'dataset':bundle.name,'drug':drug,'cell':cell,
               'generated_at_utc':datetime.now(timezone.utc).isoformat(),
               'model_sha256':hashlib.sha256((ROOT/'artifacts/model.pt').read_bytes()).hexdigest(),
               'metadata_sha256_normalized_lf':hashlib.sha256((ROOT/'artifacts/metadata.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
               'ranking_basis':'Ascending predicted IC50 within the same dataset/cell line; not a clinical recommendation',
               'patient_health_status':'Unavailable — no patient records in the dataset',
               'feature_shap_status':feature_status, 'distribution_warning':shifted,
               'group_baseline_neglog10_ic50':baseline,
               'group_shap_additivity_residual':float(selected.iloc[0]['predicted_neglog10_ic50']-baseline-group_frame['shap_value'].sum()),
               'limitations':LIMITATION}
    scale_tables,scale_figures=interpretation_data(bundle,drug,float(selected.iloc[0]["predicted_neglog10_ic50"]),selected.iloc[0]["observed_neglog10_ic50"])
    tables.update(scale_tables)
    figures = make_figures(tables, bundle.y)
    figures.update(scale_figures)
    payload={'context':context, 'metrics':metrics, 'audits':audits, 'tables':tables, 'figures':figures}
    add_chart_source_tables(payload,bundle)
    payload['model_metadata']=metadata
    add_evaluation_tables(payload)
    return payload


def make_figures(tables, response):
    figures = {}
    def save(name, title, draw):
        figure = Figure(figsize=(9,4.8), layout='constrained')
        axis = figure.subplots()
        draw(axis)
        axis.set_title(title, fontsize=12, pad=16)
        figure.set_facecolor('#ffffff')
        axis.set_facecolor('#f6fafc')
        for spine in axis.spines.values():
            spine.set_color('#b7cdd8')
        stream = BytesIO()
        figure.savefig(stream, format='png', dpi=125)
        figures[name] = {'title':title, 'png':stream.getvalue()}
    with PLOT_LOCK:
        save('response_distribution','GDSC response distribution' if len(response)>20000 else 'Selected dataset response distribution',
             lambda ax:(ax.hist(response,bins=45,color='#3988a4',edgecolor='white'),ax.set_xlabel('Measured −log10(IC50)'),ax.set_ylabel('Cell-line pairs')))
        for key,label in [('saved_holdout_predictions','GDSC held-out evaluation'),('saved_ccle_external_predictions','CCLE external evaluation')]:
            if key not in tables:
                continue
            frame = tables[key]
            def agreement(ax, frame=frame):
                ax.hexbin(frame['actual_neglog10_ic50'],frame['predicted_neglog10_ic50'],gridsize=38,mincnt=1,cmap='GnBu')
                low=float(min(frame['actual_neglog10_ic50'].min(),frame['predicted_neglog10_ic50'].min()))
                high=float(max(frame['actual_neglog10_ic50'].max(),frame['predicted_neglog10_ic50'].max()))
                ax.plot([low,high],[low,high],color='#c36b54',linestyle='--',label='Exact agreement')
                ax.set_xlabel('Observed −log10(IC50)');ax.set_ylabel('Predicted −log10(IC50)');ax.legend()
            save(key+'_agreement',label+' · observed vs predicted',agreement)
            if key=='saved_holdout_predictions':
                errors = frame['predicted_neglog10_ic50']-frame['actual_neglog10_ic50']
                save('holdout_residuals','Held-out prediction errors',lambda ax:(ax.hist(errors,bins=40,color='#3988a4'),ax.axvline(0,color='#c36b54'),ax.set_xlabel('Predicted − observed (log scale)'),ax.set_ylabel('Pairs')))
                mae=frame.assign(error=errors.abs()).groupby('drug')['error'].mean().nlargest(15).sort_values()
                save('holdout_drug_errors','Largest per-drug holdout mean absolute errors',lambda ax:(ax.barh(mae.index,mae,color='#3988a4'),ax.set_xlabel('MAE on −log10(IC50) scale')))
        ranking=pd.concat([tables['cell_line_drug_rankings'].head(12),tables['selected_pair']]).drop_duplicates('drug').sort_values('research_rank',ascending=False)
        def comparison(ax):
            ax.barh(ranking['drug'],ranking['predicted_ic50_uM_assuming_source_unit'],color=['#c36b54' if flag else '#218c86' for flag in ranking['selected_drug']])
            measured=ranking.dropna(subset=['observed_ic50_uM_assuming_source_unit'])
            ax.scatter(measured['observed_ic50_uM_assuming_source_unit'],measured['drug'],marker='D',color='#172f45',s=30,label='Observed where available',zorder=3)
            ax.set_xscale('log');ax.set_xlabel('IC50 (μM assuming source unit; log axis)');ax.legend(fontsize=8)
        save('candidate_comparison','Lowest predicted IC50 candidates and reference drug · exploratory',comparison)
        groups=tables['local_group_shap'].sort_values('shap_value')
        save('local_group_shap','Selected pair · exact five-group SHAP',lambda ax:(ax.barh(groups['feature_group'],groups['shap_value'],color=['#218c86' if v>=0 else '#c36b54' for v in groups['shap_value']]),ax.axvline(0,color='#172f45'),ax.set_xlabel('Contribution to predicted −log10(IC50)')))
        pathways=tables['local_pathway_shap'].head(15).sort_values('shap_value')
        if not pathways.empty:
            save('local_pathway_shap','Selected pair · largest approximate pathway SHAP contributions',lambda ax:(ax.barh(pathways['feature'].str.replace('__',': ',regex=False),pathways['shap_value'],color=['#218c86' if v>=0 else '#c36b54' for v in pathways['shap_value']]),ax.axvline(0,color='#172f45'),ax.tick_params(axis='y',labelsize=7),ax.set_xlabel('Model contribution; not a causal pathway effect')))
        for path in sorted((OUTPUTS/'figures').glob('*.png')):
            figures['saved_'+path.stem]={'title':'Saved evaluation figure · '+path.stem.replace('_',' '),'png':path.read_bytes()}
    return figures




def add_holdout_view(payload, view):
    """Include the visible interactive filter as a static report chart; keep all CSV rows."""
    frame=payload['tables'].get('saved_holdout_predictions')
    if frame is None:
        return
    visible=frame if view=='All drugs' else frame[frame['drug']==view]
    if visible.empty:
        raise ValueError('The selected holdout view has no rows.')
    with PLOT_LOCK:
        figure=Figure(figsize=(9,4.8),layout='constrained')
        axis=figure.subplots()
        axis.scatter(visible['actual_neglog10_ic50'],visible['predicted_neglog10_ic50'],s=12,alpha=.45,color='#218c86')
        axis.set_xlabel('Observed −log10(IC50)');axis.set_ylabel('Predicted −log10(IC50)')
        title='Current held-out view · '+view
        axis.set_title(title)
        stream=BytesIO();figure.savefig(stream,format='png',dpi=125)
        payload['figures']['interactive_holdout_view']={'title':title,'png':stream.getvalue()}
    payload['context']['interactive_holdout_filter']=view
    payload['context']['interactive_holdout_rows']=len(visible)


def readable_selected_pair(selected):
    row=selected.iloc[0]
    labels=[('Dataset','dataset'),('Drug','drug'),('Cancer cell line','cell'),
            ('Predicted −log10(IC50)','predicted_neglog10_ic50'),('Observed −log10(IC50)','observed_neglog10_ic50'),
            ('Predicted IC50 (μM, assuming source unit)','predicted_ic50_uM_assuming_source_unit'),
            ('Observed IC50 (μM, assuming source unit)','observed_ic50_uM_assuming_source_unit'),
            ('Absolute log-scale error','absolute_log_error'),('Concentration fold error','concentration_fold_error'),
            ('Research rank','research_rank'),('Patient health status','patient_health_status'),('Evidence scope','evidence_scope')]
    return pd.DataFrame([{'Measure':label,'Value':'Unavailable' if pd.isna(row[key]) else (f'{row[key]:.6g}' if isinstance(row[key], (int,float,np.number)) else str(row[key]))} for label,key in labels])


def readable_rankings(frame):
    labels={'research_rank':'Research rank','drug':'Drug',
            'predicted_ic50_uM_assuming_source_unit':'Predicted IC50 (μM*)',
            'observed_ic50_uM_assuming_source_unit':'Observed IC50 (μM*)',
            'absolute_log_error':'Absolute log error','patient_health_status':'Patient health status'}
    return frame[list(labels)].rename(columns=labels)


def add_evaluation_tables(payload):
    """Define once: on-screen tables and complete exports cannot drift apart."""
    tables=payload['tables']
    tables['evaluation_selected_pair']=readable_selected_pair(tables['selected_pair'])
    tables['evaluation_drug_comparisons']=readable_rankings(tables['cell_line_drug_rankings'])
    rows=[]
    for scope,metrics in [('GDSC holdout',payload['metrics'].get('holdout',{})),('CCLE external',payload['metrics'].get('ccle_external',{}))]:
        rows.append({'Scope':scope,'MAE':metrics.get('mae_neglog10_ic50'),'RMSE':metrics.get('rmse_neglog10_ic50'),'R²':metrics.get('r2'),'Pairs':metrics.get('n_test')})
    tables['evaluation_metric_summary']=pd.DataFrame(rows)
    tables['evaluation_interpretation_notes']=pd.DataFrame([
        {'Item':'Comparison basis','Value':'Lower predicted IC50 ranks research candidates; it does not establish clinical efficacy or safety.'},
        {'Item':'Patient health, safe dose, duration, risk and cost','Value':'Unavailable from these cell-line data; no patient treatment recommendation.'},
        {'Item':'Chart values','Value':'Hover marks for exact values; move anywhere inside axes for cursor coordinates. Density-bin centers aggregate observations and are not individual patient records.'},
        {'Item':'External reliability','Value':'CCLE transfer remains poor; the historical holdout has been reused for model selection.'},
        {'Item':'Source units','Value':'μM is inherited from the source-unit assumption. A response distribution is not a dose–response curve.'},
    ])
    names=['evaluation_selected_pair','evaluation_drug_comparisons','local_group_shap','local_pathway_shap','local_feature_shap','evaluation_metric_summary','ic50_scale_reference','ic50_same_drug_context','clinical_dose_duration_cost_availability','evaluation_interpretation_notes']
    names+=sorted(name for name in tables if name.startswith('saved_'))
    tables['evaluation_table_inventory']=pd.DataFrame([{'Table':name,'Rows':len(tables[name]),'Columns':len(tables[name].columns),'Export':'Complete embedded CSV and ZIP CSV'} for name in names])

def render_report(payload):
    from .workspace_report import attach_workspace_outputs, full_table_download, section_content, TABLE_PAGER_SCRIPT
    from .interactive import build_charts, gallery_html
    if 'workspace_section_coverage' not in payload['tables']:
        attach_workspace_outputs(payload,{})
    context=payload['context'];tables=payload['tables'];selected=tables['selected_pair'].iloc[0]
    lower=tables['cell_line_drug_rankings'][tables['cell_line_drug_rankings']['predicted_ic50_uM_assuming_source_unit'] < selected['predicted_ic50_uM_assuming_source_unit']]
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PathDSP workspace research report</title><style>body{font:15px/1.6 system-ui,sans-serif;margin:40px auto;max-width:1100px;padding:0 24px;color:#18394c;background:#fff}h1,h2,h3{line-height:1.2}h2{margin-top:40px;border-bottom:2px solid #b6dadb;padding-bottom:12px}.notice{padding:18px;background:#fff2df;border-left:4px solid #c36b54}.scope{padding:18px;background:#edf7f8}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border:1px solid #d1e2e9;text-align:left}th{background:#edf7f8}.scroll{overflow-x:auto}figure{margin:28px 0;break-inside:avoid}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}@media print{body{margin:0;max-width:none}details>summary{page-break-after:avoid}img{max-height:80vh}a{color:inherit}}</style><body>']
    parts += ['<h1>PathDSP · Workspace research report</h1>',f'<p>{escape(context["dataset"])} · {escape(context["drug"])} × {escape(context["cell"])}</p>',f'<p>Analysis snapshot {escape(datetime.fromisoformat(context["generated_at_utc"]).astimezone(ZoneInfo("Asia/Kolkata")).isoformat())} (India time) · Model {escape(context["model_sha256"][:16])}</p>',f'<div class="notice">{escape(LIMITATION)}</div>']
    parts += ['<h2>Key validation statistics</h2>',tables['evaluation_metric_summary'].to_html(index=False,escape=True),'<details><summary>Evaluation table coverage · completeness audit</summary>',tables['evaluation_table_inventory'].to_html(index=False,escape=True),'</details>',tables['evaluation_interpretation_notes'].to_html(index=False,escape=True)]
    parts += ['<h2>1. Selected pair: measurement and model estimate</h2>', '<p>Observed and predicted columns are separate. Missing measurements remain blank. Patient health status is explicitly unavailable, not inferred.</p>', '<div class="scroll">'+readable_selected_pair(tables['selected_pair']).to_html(index=False,na_rep='Unavailable',escape=True)+'</div>']
    if len(lower):
        message=f'{len(lower)} drugs have a lower predicted IC50 than the selected drug for this cell line. These are candidates for further research, not evidence of better clinical treatment.'
    else:
        message='No drug in this catalogue has a lower predicted IC50 for this cell line. This ranking does not establish treatment efficacy.'
    parts += ['<h2>IC50 scale, patient applicability, duration and cost</h2>', '<p>'+escape(INTERPRETATION)+'</p>', '<p>A one-unit log-response difference is a tenfold concentration difference. No patient benefit or toxicity threshold can be inferred from a value such as 9 or 16.5 μM alone. Clinical duration and cost cannot be calculated from these cell-line data.</p>', tables['clinical_dose_duration_cost_availability'].to_html(index=False,escape=True),tables['ic50_scale_reference'].to_html(index=False,escape=True),tables['ic50_same_drug_context'].to_html(index=False,escape=True,na_rep='Unavailable')]
    parts += ['<h2>2. Comparative drug ranking</h2>',f'<p>{escape(message)}</p>','<p>All eligible drugs are included below. Ranking does not account for toxicity, dose feasibility, uncertainty or patient biology.</p>','<div class="scroll">'+readable_rankings(tables['cell_line_drug_rankings']).to_html(index=False,na_rep='Unavailable',escape=True,float_format=lambda value:f'{value:.5g}')+'</div>']
    parts += ['<h2>3. Pathway and SHAP interpretation</h2>','<p>No one-to-one drug–pathway assignment is supplied. The following are model attributions for the selected drug–cell pair. Positive SHAP raises −log10(IC50), corresponding to lower estimated IC50; negative SHAP moves it in the opposite direction. Feature SHAP is approximate; group SHAP is exact over five groups.</p>',f'<p>{escape(context["feature_shap_status"])}</p>',tables['local_group_shap'].to_html(index=False,escape=True),'<h3>Largest pathway contributions</h3>',tables['local_pathway_shap'].head(25).to_html(index=False,escape=True),f'<p>Exact group additivity residual: {context["group_shap_additivity_residual"]:.3g}.</p>']
    parts += ['<h2>4. All evaluation visualizations</h2>','<p>The active release serves the same checkpoint used for the displayed GDSC holdout and CCLE evaluation. Fit/validation/holdout membership is recorded; the holdout has been reused for model selection and is not an independent final certification. The response distribution is not a concentration–response dose curve.</p>']
    parts.append('<p>Move the pointer anywhere within a plotting area to read axis coordinates; hover marks for exact data, counts or correlations. Log axes report concentrations in their displayed units. Cursor positions are not additional measurements. These interactions work offline in this HTML; printed/PDF copies are static.</p>')
    parts.append(gallery_html(build_charts(payload)))
    parts.append('<style>.print-only{display:none}@media print{.print-only{display:block}iframe{display:none}}</style><div class="print-only">')
    for figure in payload['figures'].values():
        encoded=base64.b64encode(figure['png']).decode('ascii')
        parts.append(f'<figure><h3>{escape(figure["title"])}</h3><img alt="{escape(figure["title"])}" src="data:image/png;base64,{encoded}"></figure>')
    parts.append('</div>')
    parts += ['<h2>5. Metrics, audit evidence and reproducibility</h2>', '<p>Overall scores refer to the complete saved evaluation sets. Rankings are for the chosen cell line. These scopes must not be combined into a single accuracy claim.</p>', '<pre>'+escape(json.dumps(payload['metrics'],indent=2))+'</pre>']
    for name,audit in payload['audits'].items():
        parts.append(f'<details><summary>{escape(name)}</summary><pre>{escape(json.dumps(audit,indent=2))}</pre></details>')
    parts += ['<h2>6. Complete data tables</h2>','<p>Every row is retained in the embedded CSV download for each table and as a CSV in the evidence ZIP. Small previews keep this large report responsive; no export is limited to the preview rows.</p>']
    for name,table in tables.items():
        parts.append(full_table_download(name,table))
    parts += ['<h2>7. Context and interpretation limits</h2>','<pre>'+escape(json.dumps(context,indent=2))+'</pre>', '<p>APIs can supply curated drug–target or pathway annotations, but API keys are access credentials, not model training or evidence of accuracy. FNN is a baseline; alternative models require study-aware validation and independent test data. No patient-specific treatment conclusions can be drawn from these cell-line features.</p>']
    parts.append(section_content(payload))
    parts.append(TABLE_PAGER_SCRIPT)
    parts.append('</body></html>')
    document='\n'.join(parts)
    contents=[]
    def section_anchor(match):
        identifier='report-section-'+str(len(contents)+1)
        contents.append('<li><a href="#'+identifier+'">'+match.group(1)+'</a></li>')
        return '<h2 id="'+identifier+'">'+match.group(1)+'</h2>'
    document=re.sub(r'<h2>(.*?)</h2>',section_anchor,document)
    document=document.replace('</h1>','</h1><nav aria-label="Report contents"><strong>Report contents</strong><ul>'+''.join(contents)+'</ul></nav>',1)
    return document.encode('utf-8')


def evidence_zip(payload, report=None):
    stream=BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('evaluation_report.html',report if report is not None else render_report(payload))
        archive.writestr('context.json',json.dumps(payload['context'],indent=2))
        archive.writestr('metrics.json',json.dumps(payload['metrics'],indent=2))
        from .interactive import build_charts
        archive.writestr('workspace_results.json',json.dumps(payload.get('workspace_results',[]),indent=2,default=str))
        archive.writestr('model_metadata.json',json.dumps(payload.get('model_metadata',{}),indent=2))
        for name,chart in build_charts(payload).items():
            archive.writestr('interactive_charts/'+name+'.plotly.json',chart.to_json())
        for name,audit in payload['audits'].items():
            archive.writestr('audits/'+name+'.json',json.dumps(audit,indent=2))
        for name,table in payload['tables'].items():
            archive.writestr('tables/'+name+'.csv',table.to_csv(index=False))
        for name,figure in payload['figures'].items():
            archive.writestr('figures/'+name+'.png',figure['png'])
        archive.writestr('README.txt','Open evaluation_report.html in a browser. It is self-contained, interactive offline, and printable (printed/PDF charts are static). Tables contain all rows; the dashboard may show shorter previews. No patient health records are supplied. Rankings and SHAP are exploratory cell-line model outputs, not treatment recommendations.\n')
    return stream.getvalue()
