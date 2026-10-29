"""Assay-scale interpretation; never a dosing, patient response or cost model."""
from io import BytesIO
import math
import threading
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

PLOT_LOCK=threading.Lock()
INTERPRETATION=('IC50 is an assay concentration associated with 50% inhibition of the assay endpoint under the measured conditions. '
    'Lower IC50 can indicate greater in-vitro potency only when assay conditions and endpoints are comparable. '
    'It is not a patient dose, a probability of success, a treatment duration, a toxicity threshold or a measure of patient progress.')


def concentration_from_target(target):
    target=float(target)
    if not math.isfinite(target) or abs(target)>300:
        raise ValueError('A finite, numerically representable log-scale target is required.')
    return 10.0 ** -target


def concentration_comparison(reference, comparison):
    reference,comparison=float(reference),float(comparison)
    if not all(math.isfinite(value) and value>0 for value in (reference,comparison)):
        raise ValueError('IC50 concentrations must be finite and greater than zero.')
    ratio=reference/comparison
    if not math.isfinite(ratio) or ratio<=0:
        raise ValueError('The concentration ratio is outside the finite numeric range.')
    log_ratio=math.log10(reference)-math.log10(comparison)
    return {'reference_ic50_uM':reference,'comparison_ic50_uM':comparison,
            'reference_target':-math.log10(reference),'comparison_target':-math.log10(comparison),
            'reference_divided_by_comparison':ratio,'target_change_comparison_minus_reference':log_ratio,
            'interpretation':'Concentration comparison only; no inference about patient benefit, safety or progress'}


def clinical_availability():
    return pd.DataFrame([
        {'Question':'Patient health or treatment response','Status':'Unavailable','What is needed':'Actual patient clinical records, validated response criteria and clinician assessment; cell-line IC50 is insufficient.'},
        {'Question':'Safe dose and treatment duration','Status':'Not determined from IC50','What is needed':'A patient-specific clinician-approved regimen, exposure/pharmacokinetic evidence and toxicity monitoring.'},
        {'Question':'Toxicity, survival or risk of death','Status':'Not determined from IC50','What is needed':'Clinical safety and outcome evidence; no validated patient-risk model exists in this project.'},
        {'Question':'Treatment cost','Status':'Unavailable','What is needed':'Drug formulation, prescribed quantity and regimen, local prices, currency, care setting and coverage; no price dataset is supplied.'},
        {'Question':'Universal sensitive/resistant cutoff','Status':'Not supplied','What is needed':'A validated drug-, assay- and context-specific threshold. Do not use arbitrary common cutoffs.'},
    ])


def scale_table():
    concentrations=np.array([.001,.01,.1,1,9,16.5,100,1000])
    return pd.DataFrame({'IC50_uM_assuming_source_unit':concentrations,'IC50_nM_assuming_source_unit':concentrations*1000,
                         'negative_log10_IC50_uM':-np.log10(concentrations)})


def interpretation_data(bundle, drug, predicted_target, observed_target):
    matching=bundle.meta['drug'].astype(str).str.casefold().eq(drug.casefold()).to_numpy()
    targets=bundle.y[matching]
    concentrations=np.power(10.0,-targets.astype(float))
    concentrations=concentrations[np.isfinite(concentrations)&(concentrations>0)]
    predicted=concentration_from_target(predicted_target)
    observed=None if observed_target is None or pd.isna(observed_target) else concentration_from_target(observed_target)
    rows=[{'Measure':'Predicted IC50 (μM, assuming source unit)','Value':predicted},
          {'Measure':'Observed IC50 (μM, assuming source unit)','Value':observed},
          {'Measure':'Usable measured cell-line responses for this drug','Value':len(concentrations)}]
    if len(concentrations):
        for label,value in zip(('Minimum','25th percentile','Median','75th percentile','Maximum'),np.percentile(concentrations,[0,25,50,75,100])):
            rows.append({'Measure':label+' measured IC50 for the same drug (μM*)','Value':float(value)})
        rows.append({'Measure':'Percent of same-drug measured IC50 values at or below this prediction (descriptive, not a clinical score)','Value':float(100*np.mean(concentrations<=predicted))})
    tables={'ic50_scale_reference':scale_table(),'ic50_same_drug_context':pd.DataFrame(rows),
            'clinical_dose_duration_cost_availability':clinical_availability()}
    if observed is not None:
        tables['ic50_prediction_observation_comparison']=pd.DataFrame([concentration_comparison(observed,predicted)])
    figures={}
    def save(name,title,draw):
        fig=Figure(figsize=(9,4.5),layout='constrained');ax=fig.subplots();draw(ax);ax.set_title(title,fontsize=11,pad=16)
        stream=BytesIO();fig.savefig(stream,format='png',dpi=125);figures[name]={'title':title,'png':stream.getvalue()}
    with PLOT_LOCK:
        def ruler(ax):
            values=[predicted]+([] if observed is None else [observed])
            low=min(-3,math.floor(math.log10(min(values)))-1);high=max(3,math.ceil(math.log10(max(values)))+1)
            ax.set_xscale('log');ax.set_xlim(10.**low,10.**high);ax.set_ylim(0,1);ax.set_yticks([])
            ax.hlines(.5,10.**low,10.**high,color='#9eb9c8',linewidth=3)
            ax.scatter([predicted],[.65],s=95,color='#287f9b',label=f'Predicted {predicted:.5g} μM*',zorder=3)
            if observed is not None:ax.scatter([observed],[.35],s=75,color='#be6c42',marker='D',label=f'Observed {observed:.5g} μM*',zorder=3)
            ax.grid(axis='x',alpha=.2);ax.set_xlabel('IC50 concentration (μM*, logarithmic axis) · each decade is 10×');ax.legend(loc='upper left',fontsize=9)
        save('ic50_concentration_scale','IC50 concentration ruler · no universal clinical threshold',ruler)
        if len(concentrations):
            def distribution(ax):
                ax.hist(np.log10(concentrations),bins=25,color='#80b8c5',edgecolor='white')
                ax.axvline(math.log10(predicted),color='#287f9b',linewidth=2,label='Selected prediction')
                if observed is not None:ax.axvline(math.log10(observed),color='#be6c42',linestyle='--',linewidth=2,label='Selected observation')
                ax.set_xlabel('log10(IC50 in μM*) for this drug across usable cell lines');ax.set_ylabel('Measured cell-line pairs');ax.legend(fontsize=9)
            save('ic50_same_drug_distribution','Same-drug assay context · descriptive, not a patient-response scale',distribution)
    return tables,figures


def render_ic50_interpretation(bundle, drug, predicted_target, observed_target):
    import streamlit as st
    tables,figures=interpretation_data(bundle,drug,predicted_target,observed_target)
    st.subheader('Understand the IC50 scale')
    st.markdown(INTERPRETATION)
    st.caption('The μM unit is inherited from the assumed source transformation. There are no validated patient-response or safe-dose cutoffs in this dataset. A one-unit change in −log10(IC50) is a tenfold concentration change.')
    for figure in figures.values():st.image(figure['png'],caption=figure['title'],width='stretch')
    with st.expander('Concentration reference table and same-drug context'):
        st.dataframe(tables['ic50_scale_reference'],hide_index=True,width='stretch')
        st.dataframe(tables['ic50_same_drug_context'],hide_index=True,width='stretch')
    with st.expander('Compare two assay concentrations · not patient progress'):
        left,right=st.columns(2)
        reference=left.number_input('Reference assay IC50 (μM)',min_value=.000000001,max_value=1000000000.,value=16.5,format='%.6g',key='ic50_reference')
        comparison=right.number_input('Comparison assay IC50 (μM)',min_value=.000000001,max_value=1000000000.,value=9.,format='%.6g',key='ic50_comparison')
        result=concentration_comparison(reference,comparison)
        st.write(f"Reference/comparison concentration ratio: **{result['reference_divided_by_comparison']:.3g}×**. Target-scale change: **{result['target_change_comparison_minus_reference']:+.3f}**.")
        st.caption('Comparisons require matched experimental conditions and endpoints. This calculator cannot tell whether a patient improved, what dose to use, or how long to treat.')
    st.dataframe(tables['clinical_dose_duration_cost_availability'],hide_index=True,width='stretch')
    return tables
