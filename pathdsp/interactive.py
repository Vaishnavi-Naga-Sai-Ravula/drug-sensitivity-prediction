"""Interactive scientific charts shared by Streamlit and offline HTML reports."""
from html import escape
import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from sklearn.metrics import confusion_matrix, roc_curve


def add_chart_source_tables(payload, bundle):
    """Retain raw response/correlation data so no coordinates are inferred from pixels."""
    from .inference import dataset_resource
    tables=payload['tables']
    responses=bundle.meta.copy()
    responses['actual_neglog10_ic50']=bundle.y
    tables['selected_dataset_responses']=responses
    gdsc=dataset_resource('GDSC')
    values=gdsc.meta.copy();values['actual_neglog10_ic50']=gdsc.y
    tables['gdsc_responses']=values
    holdout=tables.get('saved_holdout_predictions')
    if holdout is not None:
        rng=np.random.default_rng(42)
        sample=holdout.iloc[rng.choice(len(holdout),size=min(2500,len(holdout)),replace=False)]
        raw=np.stack([gdsc.pair_features(row.drug,row.cell) for row in sample.itertuples()])
        ids=np.argsort(np.nanvar(raw.astype(float),axis=0))[-20:]
        names=[gdsc.feature_names[i] for i in ids]+['response_target']
        matrix=np.corrcoef(np.column_stack([raw[:,ids],sample['actual_neglog10_ic50']]),rowvar=False)
        frame=pd.DataFrame(matrix,columns=names);frame.insert(0,'feature',names)
        tables['holdout_feature_correlations']=frame
    from matplotlib.figure import Figure
    from .ic50 import PLOT_LOCK
    for source,name in [('saved_holdout_predictions','gdsc_agreement_density_bins'),('saved_ccle_external_predictions','ccle_agreement_density_bins')]:
        if source not in tables:continue
        data=tables[source]
        with PLOT_LOCK:
            figure=Figure();axis=figure.subplots()
            bins=axis.hexbin(data['actual_neglog10_ic50'],data['predicted_neglog10_ic50'],gridsize=38,mincnt=1)
            offsets=bins.get_offsets();counts=np.asarray(bins.get_array(),dtype=int)
        tables[name]=pd.DataFrame({'observed_bin_center':offsets[:,0],'predicted_bin_center':offsets[:,1],'pair_count':counts})



def _histogram(values, title, xlabel, bins=40):
    counts,edges=np.histogram(np.asarray(values,dtype=float),bins=bins)
    centers=(edges[:-1]+edges[1:])/2
    fig=go.Figure(go.Bar(x=centers,y=counts,width=np.diff(edges)*.97,
        customdata=np.column_stack([edges[:-1],edges[1:]]),marker_color='#3997a8',
        hovertemplate='Bin start: %{customdata[0]:.5g}<br>Bin end: %{customdata[1]:.5g}<br>Count: %{y}<br>Center: %{x:.5g}<extra></extra>'))
    fig.update_layout(title=title,xaxis_title=xlabel,yaxis_title='Count')
    return fig


def _bars(frame, category, value, title, xlabel):
    frame=frame.sort_values(value)
    fig=go.Figure(go.Bar(x=frame[value],y=frame[category],orientation='h',
        marker_color=['#218c86' if v>=0 else '#b96b4f' for v in frame[value]],
        hovertemplate='%{y}<br>Value: %{x:.6g}<extra></extra>'))
    fig.update_layout(title=title,xaxis_title=xlabel,yaxis_title=category.replace('_',' '),margin_l=190)
    return fig


def _agreement(frame, title):
    fig=go.Figure(go.Scattergl(x=frame['actual_neglog10_ic50'],y=frame['predicted_neglog10_ic50'],mode='markers',
        marker={'size':5,'color':'#2b8e99','opacity':.55},
        customdata=frame[['drug','cell','absolute_error']].to_numpy(),
        hovertemplate='Drug: %{customdata[0]}<br>Cell: %{customdata[1]}<br>Observed: %{x:.6f}<br>Predicted: %{y:.6f}<br>Absolute error: %{customdata[2]:.6f}<extra></extra>'))
    low=float(min(frame['actual_neglog10_ic50'].min(),frame['predicted_neglog10_ic50'].min()))
    high=float(max(frame['actual_neglog10_ic50'].max(),frame['predicted_neglog10_ic50'].max()))
    fig.add_trace(go.Scatter(x=[low,high],y=[low,high],mode='lines',line={'dash':'dash','color':'#be795b'},name='Exact agreement',hoverinfo='skip'))
    fig.update_layout(title=title,xaxis_title='Observed −log10(IC50)',yaxis_title='Predicted −log10(IC50)')
    return fig



EVALUATION_CHART_NAMES=('interactive_holdout_view','gdsc_agreement_density','ccle_agreement_density')


def evaluation_charts(payload,dark=False):
    charts=build_charts(payload,dark)
    return {name:charts[name] for name in EVALUATION_CHART_NAMES}


def _density(frame,source,title):
    fig=go.Figure(go.Scatter(x=frame['observed_bin_center'],y=frame['predicted_bin_center'],mode='markers',
        showlegend=False,marker={'symbol':'hexagon','size':14,'color':frame['pair_count'],'colorscale':[[0,'#f7fcf0'],[.2,'#ccebc5'],[.5,'#7bccc4'],[.8,'#43a2ca'],[1,'#08589e']],'showscale':True,'colorbar':{'title':'Pairs per bin'}},
        customdata=frame['pair_count'],hovertemplate='Observed bin center: %{x:.6f}<br>Predicted bin center: %{y:.6f}<br>Pairs in bin: %{customdata}<extra></extra>'))
    low=float(min(source['actual_neglog10_ic50'].min(),source['predicted_neglog10_ic50'].min()))
    high=float(max(source['actual_neglog10_ic50'].max(),source['predicted_neglog10_ic50'].max()))
    fig.add_trace(go.Scatter(x=[low,high],y=[low,high],mode='lines',name='Exact agreement',line={'color':'#bb7b61','dash':'dash'},hoverinfo='skip'))
    fig.update_layout(title=title,xaxis={'title':'Observed −log10(IC50)','fixedrange':True},yaxis={'title':'Predicted −log10(IC50)','fixedrange':True},meta={'hexbin_grid_size':38},legend={'x':.02,'y':.98,'xanchor':'left','yanchor':'top'})
    return fig

def ic50_charts(response,drug,predicted_target,observed_target):
    charts={}
    predicted=10.**-float(predicted_target);actual=None if observed_target is None or pd.isna(observed_target) else 10.**-float(observed_target)
    fig=go.Figure(go.Scatter(x=[predicted],y=['Predicted'],mode='markers',marker={'size':14,'color':'#398ca8'},name='Predicted',hovertemplate='Predicted IC50: %{x:.6g} μM*<extra></extra>'))
    if pd.notna(actual):fig.add_trace(go.Scatter(x=[actual],y=['Observed'],mode='markers',marker={'size':14,'symbol':'diamond','color':'#bc7956'},name='Observed',hovertemplate='Observed IC50: %{x:.6g} μM*<extra></extra>'))
    fig.update_layout(title='IC50 concentration ruler · no clinical threshold',xaxis={'type':'log','title':'IC50 (μM*, logarithmic axis)'},yaxis_title='Estimate / measurement')
    values=[predicted]+([] if not pd.notna(actual) else [float(actual)])
    fig.update_xaxes(range=[min(-3,float(np.floor(np.log10(min(values))))-1),max(3,float(np.ceil(np.log10(max(values))))+1)])
    charts['ic50_concentration_scale']=fig
    same=response[response['drug']==drug]
    fig=_histogram(-same['actual_neglog10_ic50'],'Same-drug assay distribution · not patient progress','log10(IC50 in μM*)',25)
    fig.add_vline(x=-float(predicted_target),line_color='#398ca8',annotation_text='Predicted')
    if pd.notna(actual):fig.add_vline(x=-float(observed_target),line_color='#bc7956',annotation_text='Observed')
    charts['ic50_same_drug_distribution']=fig
    return charts


def attribution_chart(frame,category,value,title):
    return _bars(frame,category,value,title,'Contribution / SHAP magnitude in model-response units')


def build_charts(payload, dark=False):
    tables=payload['tables'];context=payload['context'];charts={}
    selected=tables['selected_pair'].iloc[0]
    for key,table,title in [('response_distribution','selected_dataset_responses',context['dataset']+' response distribution'),('saved_target_distribution','gdsc_responses','GDSC response distribution')]:
        charts[key]=_histogram(tables[table]['actual_neglog10_ic50'],title,'Measured −log10(IC50)',45 if key=='response_distribution' else 40)
    for key,label in [('saved_holdout_predictions','GDSC held-out evaluation'),('saved_ccle_external_predictions','CCLE external evaluation')]:
        if key not in tables:continue
        charts[key+'_agreement']=_agreement(tables[key],label+' · observed vs predicted')
    holdout=tables.get('saved_holdout_predictions')
    if holdout is not None:
        charts['saved_actual_vs_predicted']=_agreement(holdout,'Observed vs predicted response · GDSC holdout')
        errors=holdout['predicted_neglog10_ic50']-holdout['actual_neglog10_ic50']
        charts['holdout_residuals']=_histogram(errors,'Held-out prediction errors','Predicted − observed (log scale)')
        mae=holdout.assign(mean_absolute_error=errors.abs()).groupby('drug',as_index=False)['mean_absolute_error'].mean().nlargest(15,'mean_absolute_error')
        charts['holdout_drug_errors']=_bars(mae,'drug','mean_absolute_error','Largest per-drug holdout mean absolute errors','MAE on −log10(IC50) scale')
        # Original saved diagnostics use observed-minus-predicted; preserve that sign.
        residual=-errors;hist=_histogram(residual,'','Observed − predicted (log scale)',35)
        fig=make_subplots(rows=1,cols=2,subplot_titles=['Residual distribution','Residuals vs predicted'])
        fig.add_trace(hist.data[0],row=1,col=1)
        fig.add_trace(go.Scattergl(x=holdout['predicted_neglog10_ic50'],y=residual,mode='markers',marker={'size':4,'opacity':.5,'color':'#3997a8'},hovertemplate='Prediction: %{x:.6f}<br>Observed − predicted: %{y:.6f}<extra></extra>'),row=1,col=2)
        fig.update_xaxes(title_text='Observed − predicted',row=1,col=1);fig.update_yaxes(title_text='Count',row=1,col=1)
        fig.update_xaxes(title_text='Predicted −log10(IC50)',row=1,col=2);fig.update_yaxes(title_text='Observed − predicted',row=1,col=2)
        fig.update_layout(title='Residual diagnostics · GDSC holdout',showlegend=False)
        charts['saved_residual_diagnostics']=fig
        threshold=payload['metrics']['holdout']['exploratory_threshold_neglog10_ic50']
        truth=holdout['actual_neglog10_ic50']>=threshold;predicted=holdout['predicted_neglog10_ic50']>=threshold
        cm=confusion_matrix(truth,predicted,labels=[False,True]);labels=['Below training median','At/above training median']
        fig=go.Figure(go.Heatmap(z=cm,x=labels,y=labels,colorscale='Blues',hovertemplate='Predicted: %{x}<br>Observed: %{y}<br>Pairs: %{z}<extra></extra>',text=cm,texttemplate='%{text}'))
        fig.update_layout(title='Exploratory confusion matrix · not clinical classes',xaxis_title='Predicted label',yaxis_title='Observed label')
        charts['saved_confusion_matrix']=fig
        fpr,tpr,cutoffs=roc_curve(truth,holdout['predicted_neglog10_ic50'])
        fig=go.Figure(go.Scatter(x=fpr,y=tpr,mode='lines',customdata=[str(v) for v in cutoffs],line_color='#218c86',hovertemplate='False positive rate: %{x:.6f}<br>True positive rate: %{y:.6f}<br>Score threshold: %{customdata}<extra></extra>'))
        fig.update_layout(title='Exploratory ROC curve · training-median labels',xaxis_title='False positive rate',yaxis_title='True positive rate')
        charts['saved_roc_curve']=fig
        view=context.get('interactive_holdout_filter','All drugs')
        frame=holdout if view=='All drugs' else holdout[holdout['drug']==view]
        charts['interactive_holdout_view']=_agreement(frame,'Current held-out view · '+view)
        charts['interactive_holdout_view'].data[0].marker.update(color=frame['absolute_error'],colorscale='GnBu',showscale=True,colorbar={'title':'Absolute error'})
    for name,table,source,title in [('gdsc_agreement_density','gdsc_agreement_density_bins','saved_holdout_predictions','GDSC held-out evaluation · observed vs predicted'),('ccle_agreement_density','ccle_agreement_density_bins','saved_ccle_external_predictions','CCLE external evaluation · observed vs predicted')]:
        if table in tables:charts[name]=_density(tables[table],tables[source],title)
    ranking=pd.concat([tables['cell_line_drug_rankings'].head(12),tables['selected_pair']]).drop_duplicates('drug').sort_values('research_rank',ascending=False)
    fig=go.Figure(go.Bar(x=ranking['predicted_ic50_uM_assuming_source_unit'],y=ranking['drug'],orientation='h',name='Predicted IC50',marker_color=['#b96b4f' if v else '#218c86' for v in ranking['selected_drug']],hovertemplate='Drug: %{y}<br>Predicted IC50: %{x:.6g} μM*<extra></extra>'))
    observed=ranking.dropna(subset=['observed_ic50_uM_assuming_source_unit'])
    fig.add_trace(go.Scatter(x=observed['observed_ic50_uM_assuming_source_unit'],y=observed['drug'],mode='markers',marker={'symbol':'diamond','color':'#879bc4','size':9},name='Observed IC50',hovertemplate='Drug: %{y}<br>Observed IC50: %{x:.6g} μM*<extra></extra>'))
    fig.update_layout(title='Research candidates and reference drug · not treatment advice',xaxis={'type':'log','title':'IC50 (μM*, logarithmic axis)'},margin_l=170)
    charts['candidate_comparison']=fig
    for key,title in [('local_group_shap','Selected pair · exact group SHAP'),('local_pathway_shap','Selected pair · approximate pathway SHAP'),('local_feature_shap','Selected pair · approximate feature SHAP')]:
        frame=tables[key];category='feature_group' if key=='local_group_shap' else 'feature'
        if not frame.empty:charts[key]=_bars(frame.head(15 if key=='local_pathway_shap' else 25),category,'shap_value',title,'Contribution to −log10(IC50)')
    history=tables.get('saved_training_history')
    if history is not None:
        fig=go.Figure()
        for column,label in [('train_rmse','Training'),('valid_rmse','Validation')]:
            fig.add_trace(go.Scatter(x=history['epoch'],y=history[column],mode='lines+markers',name=label,hovertemplate=label+'<br>Epoch: %{x}<br>RMSE: %{y:.6f}<extra></extra>'))
        fig.update_layout(title='FNN training history',xaxis_title='Epoch',yaxis_title='RMSE on −log10(IC50)')
        charts['saved_training_curve']=fig
    for key,table,category,value,title in [('saved_global_group_shap','saved_global_group_shap','feature_group','mean_absolute_group_shap','Global mean absolute group SHAP'),('saved_feature_shap_global','saved_global_feature_shap','feature','mean_absolute_shap','Global mean absolute feature SHAP · top 25')]:
        if table in tables:charts[key]=_bars(tables[table].head(25),category,value,title,'Mean absolute SHAP')
    if 'holdout_feature_correlations' in tables:
        correlation=tables['holdout_feature_correlations'];names=correlation['feature'].tolist()
        fig=go.Figure(go.Heatmap(z=correlation.drop(columns='feature').to_numpy(),x=names,y=names,zmin=-1,zmax=1,colorscale='RdBu',hovertemplate='X feature: %{x}<br>Y feature: %{y}<br>Pearson r: %{z:.6f}<extra></extra>'))
        fig.update_layout(title='Holdout feature/response correlations · not causation',height=620,margin={'l':180,'b':170},xaxis={'tickfont':{'size':8}},yaxis={'tickfont':{'size':8}})
        charts['saved_correlation_heatmap']=fig
    charts.update(ic50_charts(tables['selected_dataset_responses'],context['drug'],selected['predicted_neglog10_ic50'],selected['observed_neglog10_ic50']))
    # Generated outputs from other workspace sections are separate, labelled snapshots.
    for key,frame in tables.items():
        if key.startswith('workspace_') and key.endswith('_prediction') and 'predicted_ic50_uM' in frame:
            row=frame.iloc[0]
            fig=go.Figure(go.Scatter(x=[row['predicted_ic50_uM']],y=['Predicted'],mode='markers',name='Predicted',marker={'size':14},hovertemplate='Predicted IC50: %{x:.6g} μM*<extra></extra>'))
            if pd.notna(row.get('actual_ic50_uM')):
                fig.add_trace(go.Scatter(x=[row['actual_ic50_uM']],y=['Observed'],mode='markers',name='Observed',marker={'size':14,'symbol':'diamond'},hovertemplate='Observed IC50: %{x:.6g} μM*<extra></extra>'))
            fig.update_layout(title='Predict snapshot · '+str(row['drug'])+' × '+str(row['cell']),xaxis={'type':'log','title':'IC50 (μM*, logarithmic axis)'},yaxis_title='Estimate / measurement')
            charts[key+'_ic50']=fig
            source=tables['gdsc_responses'] if row['dataset']=='GDSC' else tables.get('saved_ccle_external_predictions')
            if source is not None:
                same=source[source['drug']==row['drug']]
                charts[key+'_distribution']=_histogram(-same['actual_neglog10_ic50'],'Predict snapshot · same-drug assay distribution','log10(IC50 in μM*)',25)
        if key.endswith('_batch_predictions') and 'predicted_neglog10_ic50' in frame:
            fig=go.Figure(go.Scatter(x=np.arange(1,len(frame)+1),y=frame['predicted_neglog10_ic50'],mode='markers',hovertemplate='Input row: %{x}<br>Predicted −log10(IC50): %{y:.6f}<extra></extra>'))
            fig.update_layout(title='Predict · generated batch results',xaxis_title='Input row (1-based)',yaxis_title='Predicted −log10(IC50)');charts[key]=fig
        if key.startswith('workspace_') and key.endswith('_shap') and 'shap_value' in frame:
            category='feature' if 'feature' in frame else 'feature_group'
            top=frame.assign(magnitude=frame.shap_value.abs()).nlargest(25,'magnitude')
            charts[key]=_bars(top,category,'shap_value',key.replace('_',' ').title(),'Contribution to −log10(IC50)')
    for fig in charts.values():
        fig.update_layout(template='plotly_dark' if dark else 'plotly_white',hovermode='closest',height=fig.layout.height or 460,autosize=True,margin_t=65,margin_b=75,font_family='system-ui',paper_bgcolor='#102b3b' if dark else '#ffffff',uirevision='preserve')
        fig.update_xaxes(showspikes=True,spikemode='across',spikesnap='cursor');fig.update_yaxes(showspikes=True,spikemode='across',spikesnap='cursor')
    return charts


GALLERY_STYLE='''<style>.interactive-card{border:1px solid #8ba9bd55;border-radius:16px;padding:12px;margin:18px 0;break-inside:avoid}.plot-shell{position:relative}.plot-area{width:100%;min-height:460px}.cursor-readout{display:block;min-height:26px;padding:8px 12px;font:13px/1.4 system-ui;background:#eaf4f7;color:#15394b;border-radius:8px;overflow-wrap:anywhere}.crosshair{position:absolute;pointer-events:none;border-color:#98a7bd88;display:none;z-index:2}.crosshair-v{border-left:1px dashed}.crosshair-h{border-top:1px dashed}.dark .cursor-readout{background:#16384b;color:#deedf5}@media print{.interactive-gallery{display:none!important}}</style>'''

GALLERY_SCRIPT=r'''
<script>
(function(){
 const specs=JSON.parse(document.getElementById('chart-specifications').textContent);
 window.pathdspCharts=specs;
 window.pathdspPlotsReady=Promise.all(specs.map((spec,index)=>{
  const graph=document.getElementById('chart-'+index),shell=graph.parentElement;
  const readout=shell.querySelector('.cursor-readout'),vertical=shell.querySelector('.crosshair-v'),horizontal=shell.querySelector('.crosshair-h');
  const mobile=graph.clientWidth<650;
  if(mobile){
   const layout=spec.figure.layout,words=(layout.title?.text||'').split(' '),lines=[];let line='';
   for(const word of words){if((line+' '+word).length>36){lines.push(line);line=word}else{line+=(line?' ':'')+word}}if(line)lines.push(line);
   layout.title={...layout.title,text:lines.join('<br>'),font:{size:12}};
   layout.margin={...layout.margin,l:Math.min(layout.margin?.l||80,90),r:20,b:110,t:80};
   layout.legend={...layout.legend,orientation:'h',x:0,y:-.3};
  }
  const config={responsive:true,scrollZoom:false,displaylogo:false};
  if(mobile)config.modeBarButtonsToRemove=['select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d'];
  return Plotly.newPlot(graph,spec.figure.data,spec.figure.layout,config).then(async()=>{
   if(spec.figure.layout.meta?.hexbin_grid_size){await Plotly.restyle(graph,{'marker.size':Math.max(5,Math.min(32,graph._fullLayout.xaxis._length/39*1.15))},[0])}
   if(mobile){
    const update={};
    for(const [name,axis] of Object.entries(graph._fullLayout)){
     if(/^[xy]axis\d*$/.test(name)&&axis.type==='category'){
      update[name+'.tickmode']='array';update[name+'.tickvals']=axis._categories;
      update[name+'.ticktext']=axis._categories.map(value=>String(value).length>15?String(value).slice(0,14)+'…':String(value));
     }
    }
    if(Object.keys(update).length)await Plotly.relayout(graph,update);
   }
   function format(value){return typeof value==='number'?Number(value.toPrecision(8)).toString():String(value)}
   graph.addEventListener('pointermove',event=>{
    const box=graph.getBoundingClientRect(),layout=graph._fullLayout;
    for(const subplot of (layout._subplots.cartesian||[])){
     const match=/^(x\d*)(y\d*)$/.exec(subplot);if(!match)continue;
     const xaxis=layout['xaxis'+match[1].slice(1)],yaxis=layout['yaxis'+match[2].slice(1)];
     if(!xaxis||!yaxis)continue;
     const px=event.clientX-box.left-xaxis._offset,py=event.clientY-box.top-yaxis._offset;
     if(px<0||py<0||px>xaxis._length||py>yaxis._length)continue;
     const x=xaxis.p2d(px),y=yaxis.p2d(py);
     readout.textContent='Cursor coordinates — x: '+format(x)+'; y: '+format(y)+' (axis units; cursor position, not a new measurement)';
     readout.dataset.x=String(x);readout.dataset.y=String(y);readout.dataset.subplot=subplot;
     vertical.style.cssText='display:block;left:'+(xaxis._offset+px)+'px;top:'+yaxis._offset+'px;height:'+yaxis._length+'px';
     horizontal.style.cssText='display:block;left:'+xaxis._offset+'px;top:'+(yaxis._offset+py)+'px;width:'+xaxis._length+'px';
     return;
    }
    vertical.style.display='none';horizontal.style.display='none';readout.textContent='Move inside the plotting axes for cursor coordinates.';delete readout.dataset.x;delete readout.dataset.y;
   });
   graph.addEventListener('pointerleave',()=>{vertical.style.display='none';horizontal.style.display='none'});
  });
 }));
})();
</script>'''


def gallery_html(charts, *, standalone=False, dark=False):
    """Bundle JS once; HTML works with no internet or external font/CDN requests."""
    sections=[];specs=[]
    for index,(name,fig) in enumerate(charts.items()):
        sections.append(f'<section class="interactive-card" data-chart="{escape(name)}"><h3>{escape(fig.layout.title.text or name)}</h3><div class="plot-shell"><div class="plot-area" id="chart-{index}" tabindex="0"></div><div class="crosshair crosshair-v"></div><div class="crosshair crosshair-h"></div><output class="cursor-readout" aria-label="Cursor coordinates">Move over the plot for coordinates; hover a data mark for exact recorded values.</output></div></section>')
        specs.append({'name':name,'figure':json.loads(fig.to_json())})
    encoded=json.dumps(specs,ensure_ascii=True,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    content=GALLERY_STYLE+'<div class="interactive-gallery '+('dark' if dark else '')+'">'+''.join(sections)+'</div>'
    content+='<script>'+get_plotlyjs()+'</script><script type="application/json" id="chart-specifications">'+encoded+'</script>'+GALLERY_SCRIPT
    if standalone:
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><body style="margin:0;font-family:system-ui;background:'+('#102b3b;color:#deedf5' if dark else '#fff;color:#17394b')+'">'+content+'</body></html>'
    return content


def show_gallery(charts, dark=False):
    import streamlit as st
    for fig in charts.values():
        fig.update_layout(template='plotly_dark' if dark else 'plotly_white',height=fig.layout.height or 460,paper_bgcolor='#102b3b' if dark else '#fff')
    html=gallery_html(charts,standalone=True,dark=dark)
    if hasattr(st,'iframe'):st.iframe(html,height='content',alt='Interactive scientific charts with cursor coordinates and data tooltips')
    else:
        import streamlit.components.v1 as components
        components.html(html,height=1100,scrolling=True)
