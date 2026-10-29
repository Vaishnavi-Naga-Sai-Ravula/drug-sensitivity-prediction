"""One evaluation payload powers on-screen research evidence and all downloads."""
from pathlib import Path
import hashlib
import json
import numpy as np
import streamlit as st
from .evaluation import build_evaluation, render_report, evidence_zip, readable_selected_pair, readable_rankings, add_holdout_view, LIMITATION

ROOT=Path(__file__).resolve().parents[1]

@st.cache_data(max_entries=4, ttl=900, show_spinner=False)
def evaluation_payload(dataset, drug, cell, fingerprint, _bundle, _model, _metadata):
    return build_evaluation(_bundle, _model, _metadata, drug, cell)


def render_evaluation(get_dataset, get_model):
    st.title('Evaluation & research comparisons')
    st.caption('Compare model evidence for a chosen cell line, inspect pathway attributions and export the complete record.')
    prior=st.session_state.get('prediction_result',{}).get('key')
    if prior and prior != st.session_state.get('_evaluation_prior_pair'):
        st.session_state['evaluation_dataset'],st.session_state['evaluation_drug'],st.session_state['evaluation_cell']=prior
        st.session_state['_evaluation_prior_pair']=prior
    source=st.selectbox('Evaluation dataset',['GDSC','CCLE'],key='evaluation_dataset')
    try:
        bundle=get_dataset(source)
        model,metadata=get_model(tuple(bundle.feature_names))
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f'Evaluation model or data unavailable: {exc}')
        return
    drugs=sorted(bundle.drug_ids,key=str.casefold);cells=sorted(bundle.cell_ids,key=str.casefold)
    for key,options in [('evaluation_drug',drugs),('evaluation_cell',cells)]:
        if st.session_state.get(key) not in options:
            st.session_state[key]=options[0]
    left,right=st.columns(2)
    drug=left.selectbox('Reference drug · type to search',drugs,key='evaluation_drug')
    cell=right.selectbox('Cell line · type to search',cells,key='evaluation_cell')
    fingerprint=hashlib.sha256((ROOT/'artifacts/manifest.json').read_bytes()).hexdigest()
    key=(source,drug,cell,fingerprint)
    try:
        with st.spinner('Comparing catalogue drugs and preparing pathway explanations…'):
            payload=evaluation_payload(source,drug,cell,fingerprint,bundle,model,metadata)
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f'Evaluation could not complete: {exc}')
        return
    tables=payload['tables'];context=payload['context'];selected=tables['selected_pair'].iloc[0]
    st.warning(LIMITATION)
    st.subheader('Selected pair · estimate versus measurement')
    a,b,c,d=st.columns(4)
    a.metric('Predicted IC50 (μM*)',f"{selected['predicted_ic50_uM_assuming_source_unit']:.4g}")
    actual=selected['observed_ic50_uM_assuming_source_unit']
    b.metric('Observed IC50 (μM*)','Unavailable' if np.isnan(actual) else f'{actual:.4g}')
    error=selected['absolute_log_error']
    c.metric('Absolute log-scale error','Unavailable' if np.isnan(error) else f'{error:.3f}')
    d.metric('Research rank',f"{int(selected['research_rank'])} / {len(tables['cell_line_drug_rankings'])}")
    st.caption('*IC50 conversion assumes the source target used μM. Patient health status: unavailable; this dataset contains cell lines, not patients.')
    st.dataframe(readable_selected_pair(tables['selected_pair']),hide_index=True,width='stretch')
    st.subheader('IC50 scale and applicability')
    from .ic50 import INTERPRETATION
    st.markdown(INTERPRETATION)
    for name in ('ic50_concentration_scale','ic50_same_drug_distribution'):
        st.image(payload['figures'][name]['png'],caption=payload['figures'][name]['title'],width='stretch')
    st.dataframe(tables['ic50_same_drug_context'],hide_index=True,width='stretch')
    st.dataframe(tables['clinical_dose_duration_cost_availability'],hide_index=True,width='stretch')
    st.subheader('Drug comparison for this cell line')
    lower=tables['cell_line_drug_rankings'][tables['cell_line_drug_rankings']['predicted_ic50_uM_assuming_source_unit'] < selected['predicted_ic50_uM_assuming_source_unit']]
    if len(lower):
        best=lower.iloc[0]
        st.info(f"Research candidate: {best['drug']} has a lower model-estimated IC50 ({best['predicted_ic50_uM_assuming_source_unit']:.4g} μM*) than {drug} for {cell}. {len(lower)} catalogue drugs rank lower by predicted IC50. This does not establish greater clinical efficacy or safety.")
    else:
        st.info('No catalogue drug has a lower predicted IC50 for this cell line. This is not evidence that the selected drug is a suitable treatment.')
    st.image(payload['figures']['candidate_comparison']['png'],caption=payload['figures']['candidate_comparison']['title'],width='stretch')
    st.dataframe(readable_rankings(tables['cell_line_drug_rankings']),hide_index=True,width='stretch')
    st.download_button('Download all drug comparisons (CSV)',tables['cell_line_drug_rankings'].to_csv(index=False),'cell_line_drug_comparisons.csv','text/csv')
    st.subheader('Pathway explanations for the selected pair')
    st.caption('A pathway is not assigned one drug. These are model contributions for the selected drug/cell pair. Group SHAP is exact over five groups; individual pathway SHAP is approximate and does not establish biological causation.')
    st.image(payload['figures']['local_group_shap']['png'],caption='Exact five-group SHAP · positive contribution raises −log10(IC50)',width='stretch')
    st.dataframe(tables['local_group_shap'],hide_index=True,width='stretch')
    if 'local_pathway_shap' in payload['figures']:
        st.image(payload['figures']['local_pathway_shap']['png'],caption='Largest approximate pathway contributions for this drug–cell pair',width='stretch')
        st.dataframe(tables['local_pathway_shap'].head(25),hide_index=True,width='stretch')
        st.caption('The report and evidence ZIP include every pathway and feature attribution, not only this preview.')
    else:
        st.warning(context['feature_shap_status'])
    st.subheader('Validation evidence · held-out and external results')
    st.caption('The same evaluated MSE-FNN checkpoint is served. Scores describe full evaluation sets, not a guarantee for one selected cell line. Historical holdout reuse prevents an independent final-certification claim.')
    metric_rows=[]
    for scope,metrics in [('GDSC holdout',payload['metrics'].get('holdout',{})),('CCLE external evaluation',payload['metrics'].get('ccle_external',{}))]:
        metric_rows.append({'scope':scope,'MAE':metrics.get('mae_neglog10_ic50'),'RMSE':metrics.get('rmse_neglog10_ic50'),'R²':metrics.get('r2'),'pairs':metrics.get('n_test')})
    st.dataframe(metric_rows,hide_index=True,width='stretch')
    view='All drugs'
    if 'saved_holdout_predictions' in tables:
        import altair as alt
        holdout=tables['saved_holdout_predictions']
        view=st.selectbox('Inspect a held-out drug',['All drugs']+sorted(holdout['drug'].unique(),key=str.casefold))
        visible=holdout if view=='All drugs' else holdout[holdout['drug']==view]
        sample=visible.sample(n=min(5000,len(visible)),random_state=42)
        chart=alt.Chart(sample).mark_circle(opacity=.6).encode(x=alt.X('actual_neglog10_ic50:Q',title='Observed −log10(IC50)'),y=alt.Y('predicted_neglog10_ic50:Q',title='Predicted −log10(IC50)'),color=alt.Color('absolute_error:Q',scale=alt.Scale(scheme='tealblues')),tooltip=['drug','cell','actual_neglog10_ic50','predicted_neglog10_ic50']).properties(height=380).interactive()
        st.altair_chart(chart,width='stretch')
        st.caption('Interactive view displays up to 5,000 points. Full data and static plots are included in the report; this filter does not alter overall metrics or exported rows.')
    st.subheader('Response distributions and diagnostic visualizations')
    st.caption('The GDSC response distribution is not a dose–response curve: raw dose-series measurements are not supplied.')
    for name in ('saved_target_distribution','response_distribution','saved_holdout_predictions_agreement','holdout_residuals','holdout_drug_errors','saved_ccle_external_predictions_agreement'):
        if name in payload['figures']:
            figure=payload['figures'][name]
            st.image(figure['png'],caption=figure['title'],width='stretch')
    with st.expander('All additional saved diagnostic figures'):
        shown={'saved_target_distribution','response_distribution','saved_holdout_predictions_agreement','holdout_residuals','holdout_drug_errors','saved_ccle_external_predictions_agreement','candidate_comparison','local_group_shap','local_pathway_shap','ic50_concentration_scale','ic50_same_drug_distribution'}
        for name,figure in payload['figures'].items():
            if name not in shown:
                st.image(figure['png'],caption=figure['title'],width='stretch')
    st.subheader('Complete data downloads')
    for name,label in [('saved_holdout_predictions','GDSC holdout predictions'),('saved_ccle_external_predictions','CCLE external predictions'),('local_group_shap','Exact local group SHAP'),('local_feature_shap','All local feature SHAP values'),('local_pathway_shap','All local pathway SHAP values')]:
        if name in tables and not tables[name].empty:
            st.download_button('Download '+label+' (CSV)',tables[name].to_csv(index=False),name+'.csv','text/csv',key='evaluation_download_'+name)
    with st.expander('Metrics, data audits and reproducibility'):
        st.json(payload['metrics'])
        st.json(payload['audits'])
        st.json(context)
    st.subheader('Download the structured evaluation report')
    st.caption('The standalone HTML contains all figures, comparisons, SHAP graphs, complete expandable data tables, audit evidence and interpretation limits. Open it in a browser; use Print → Save as PDF if needed. The evidence ZIP also contains every CSV and PNG separately. The report includes all evidence, regardless of collapsed sections or the interactive drug filter.')
    key=(*key,view)
    if st.button('Prepare complete report and evidence bundle',type='primary'):
        with st.spinner('Embedding all tables and figures…'):
            add_holdout_view(payload,view)
            report=render_report(payload)
            st.session_state['evaluation_export']={'key':key,'report':report,'zip':evidence_zip(payload,report)}
    exported=st.session_state.get('evaluation_export',{})
    if exported.get('key')==key:
        a,b=st.columns(2)
        a.download_button('Download evaluation report (HTML)',exported['report'],'pathdsp_evaluation_report.html','text/html')
        b.download_button('Download complete evidence (ZIP)',exported['zip'],'pathdsp_evaluation_evidence.zip','application/zip')
