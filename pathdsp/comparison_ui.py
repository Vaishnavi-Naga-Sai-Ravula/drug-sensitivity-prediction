"""Explicit matched-cell research comparison, never a patient recommendation."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .inference import predict_pairs, model_resource
from .interactive import show_gallery
from .workspace_report import record_output


def render_comparison(bundle,reference,cell):
    with st.expander('Compare research candidates · same cell line'):
        st.caption('Compare measured and predicted responses under the same dataset and cell line. Lower IC50 alone does not establish clinical superiority or account for toxicity or achievable exposure.')
        key='compare_drugs_'+bundle.name
        if key in st.session_state:
            st.session_state[key]=[value for value in st.session_state[key] if value in bundle.drug_ids][:8]
        drugs=st.multiselect('Drugs to compare (up to 8)',sorted(bundle.drug_ids,key=str.casefold),default=[reference] if key not in st.session_state else None,max_selections=8,key=key)
        identity=(bundle.name,cell,tuple(drugs),reference,model_resource()[2])
        if st.button('Run matched-cell comparison',disabled=not drugs):
            try:
                response=predict_pairs(bundle.name,[(drug,cell) for drug in drugs])
                st.session_state['matched_comparison']={'key':identity,'response':response}
            except (ValueError,RuntimeError,OSError) as exc:
                st.error(f'Comparison unavailable: {exc}')
                return
        saved=st.session_state.get('matched_comparison',{})
        if saved.get('key')!=identity:return
        response=saved['response']
        frame=pd.DataFrame([{key:value for key,value in row.items() if key!='evidence'} for row in response['results']])
        frame['dataset']=bundle.name
        reference_row=frame.loc[frame.drug.eq(reference)]
        frame['predicted_ic50_ratio_to_reference']=np.nan if reference_row.empty else frame.predicted_ic50_uM_assuming_source_unit/float(reference_row.iloc[0].predicted_ic50_uM_assuming_source_unit)
        frame['interval_status']='Unavailable for serving checkpoint'
        frame['clinical_superiority']='Not established'
        st.dataframe(frame,hide_index=True,width='stretch')
        fig=comparison_chart(frame)
        show_gallery({'matched_cell_comparison':fig},dark=st.session_state.get('dark_mode',False))
        st.download_button('Download matched-cell comparison',frame.to_csv(index=False),'matched_cell_comparison.csv','text/csv')
        record_output(st.session_state,'comparison',{'drug_comparison':frame},{'dataset':bundle.name,'cell':cell,'reference':reference,'model_version':response['model_version']})


def comparison_chart(frame):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=frame['drug'],y=frame['predicted_ic50_uM_assuming_source_unit'],mode='markers',name='Predicted',marker={'size':13},hovertemplate='Drug: %{x}<br>Predicted IC50: %{y:.6g} μM*<extra></extra>'))
    actual=frame.dropna(subset=['observed_neglog10_ic50'])
    fig.add_trace(go.Scatter(x=actual['drug'],y=10.**-actual['observed_neglog10_ic50'],mode='markers',name='Observed',marker={'symbol':'diamond','size':12},hovertemplate='Drug: %{x}<br>Observed IC50: %{y:.6g} μM*<extra></extra>'))
    fig.update_layout(title='Matched-cell research comparison · not treatment advice',xaxis_title='Drug',yaxis={'type':'log','title':'IC50 (μM*, conditional source unit)'})
    return fig
