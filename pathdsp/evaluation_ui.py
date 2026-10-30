"""Compact tables-and-charts evaluation with full-workspace report exports."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import streamlit as st
from .evaluation import build_evaluation, render_report, evidence_zip, readable_selected_pair, readable_rankings, add_holdout_view
from .interactive import evaluation_charts, show_gallery
from .workspace_report import attach_workspace_outputs

ROOT=Path(__file__).resolve().parents[1]

@st.cache_data(max_entries=4,ttl=900,show_spinner=False)
def evaluation_payload(dataset,drug,cell,fingerprint,_bundle,_model,_metadata):
    return build_evaluation(_bundle,_model,_metadata,drug,cell)


def render_evaluation(get_dataset,get_model):
    st.title('Evaluation')
    prior=st.session_state.get('prediction_result',{}).get('key')
    if prior and prior!=st.session_state.get('_evaluation_prior_pair'):
        st.session_state['evaluation_dataset'],st.session_state['evaluation_drug'],st.session_state['evaluation_cell']=prior
        st.session_state['_evaluation_prior_pair']=prior
    source=st.selectbox('Evaluation dataset',['GDSC','CCLE'],key='evaluation_dataset')
    try:
        bundle=get_dataset(source);model,metadata=get_model(tuple(bundle.feature_names))
    except (ValueError,RuntimeError,OSError) as exc:
        st.error(f'Evaluation model or data unavailable: {exc}');return
    drugs=sorted(bundle.drug_ids,key=str.casefold);cells=sorted(bundle.cell_ids,key=str.casefold)
    for key,options in [('evaluation_drug',drugs),('evaluation_cell',cells)]:
        if st.session_state.get(key) not in options:st.session_state[key]=options[0]
    left,right=st.columns(2)
    drug=left.selectbox('Reference drug · type to search',drugs,key='evaluation_drug')
    cell=right.selectbox('Cell line · type to search',cells,key='evaluation_cell')
    fingerprint=hashlib.sha256((ROOT/'artifacts/manifest.json').read_bytes()).hexdigest()
    try:
        with st.spinner('Preparing tables and interactive visualizations…'):
            payload=evaluation_payload(source,drug,cell,fingerprint,bundle,model,metadata)
    except (ValueError,RuntimeError,OSError) as exc:
        st.error(f'Evaluation could not complete: {exc}');return
    tables=payload['tables']
    view='All drugs'
    if 'saved_holdout_predictions' in tables:
        view=st.selectbox('Held-out chart drug filter',['All drugs']+sorted(tables['saved_holdout_predictions']['drug'].unique(),key=str.casefold))
    payload['context']['interactive_holdout_filter']=view
    payload['context']['evaluation_visible_charts']=['interactive_holdout_view','gdsc_agreement_density','ccle_agreement_density']
    selected,rankings,explanations,validation,scale=st.tabs(['Selected pair','Drug comparisons','SHAP & pathways','Validation data','IC50 scale'])
    with selected:st.dataframe(tables['evaluation_selected_pair'],hide_index=True,width='stretch')
    with rankings:st.dataframe(tables['evaluation_drug_comparisons'],hide_index=True,width='stretch')
    with explanations:
        st.dataframe(tables['local_group_shap'],hide_index=True,width='stretch')
        st.dataframe(tables['local_pathway_shap'],hide_index=True,width='stretch')
        st.dataframe(tables['local_feature_shap'],hide_index=True,width='stretch')
    with validation:
        st.dataframe(tables['evaluation_metric_summary'],hide_index=True,width='stretch')
        saved=st.selectbox('Validation table',sorted(name for name in tables if name.startswith('saved_')))
        st.dataframe(tables[saved],hide_index=True,width='stretch')
    with scale:
        st.dataframe(tables['ic50_scale_reference'],hide_index=True,width='stretch')
        st.dataframe(tables['ic50_same_drug_context'],hide_index=True,width='stretch')
        st.dataframe(tables['clinical_dose_duration_cost_availability'],hide_index=True,width='stretch')
    st.dataframe(tables['evaluation_interpretation_notes'],hide_index=True,width='stretch')
    show_gallery(evaluation_charts(payload,dark=st.session_state.get('dark_mode',False)),dark=st.session_state.get('dark_mode',False))
    csv_table=st.selectbox('Download complete table',sorted(tables))
    st.download_button('Download selected table (CSV)',tables[csv_table].to_csv(index=False),csv_table+'.csv','text/csv')
    snapshots=st.session_state.get('workspace_outputs',{})
    calculator=None
    if 'ic50_reference' in st.session_state and 'ic50_comparison' in st.session_state:
        calculator=(st.session_state['ic50_reference'],st.session_state['ic50_comparison'])
    snapshot_token=hashlib.sha256(json.dumps([sorted(snapshots),calculator],default=str).encode()).hexdigest()
    key=(source,drug,cell,fingerprint,view,snapshot_token)
    if st.button('Prepare complete report and evidence bundle',type='primary'):
        with st.spinner('Embedding interactive charts and all generated workspace results…'):
            add_holdout_view(payload,view)
            attach_workspace_outputs(payload,snapshots,calculator)
            report=render_report(payload)
            st.session_state['evaluation_export']={'key':key,'report':report,'zip':evidence_zip(payload,report)}
    exported=st.session_state.get('evaluation_export',{})
    if exported.get('key')==key:
        left,right=st.columns(2)
        left.download_button('Download evaluation report (HTML)',exported['report'],'pathdsp_evaluation_report.html','text/html')
        right.download_button('Download complete evidence (ZIP)',exported['zip'],'pathdsp_evaluation_evidence.zip','application/zip')
