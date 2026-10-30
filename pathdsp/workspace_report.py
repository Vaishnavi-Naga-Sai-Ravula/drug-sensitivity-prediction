"""Capture generated session results without executing analysis that was never requested."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from html import escape
import base64
import re
from .section_content import OVERVIEW_CARDS, OVERVIEW_BLOCKS, METHODS_SCIENCE
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def record_output(state, section, tables, context=None, text=None):
    context=context or {}
    digest=hashlib.sha256(json.dumps(context,sort_keys=True,default=str).encode())
    for name,table in sorted(tables.items()):
        digest.update(name.encode());digest.update(table.to_csv(index=False).encode())
    key=section+'_'+digest.hexdigest()[:12]
    state.setdefault('workspace_outputs',{})[key]={'section':section,'context':context,
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'tables':{name:table.copy(deep=True) for name,table in tables.items()},'text':text or ''}


def attach_workspace_outputs(payload, snapshots, calculator=None):
    records=[];sections=set()
    for key,snapshot in snapshots.items():
        sections.add(snapshot['section'])
        for name,table in snapshot['tables'].items():
            payload['tables']['workspace_'+key+'_'+name]=table.copy(deep=True)
        records.append({name:snapshot[name] for name in ('section','context','generated_at_utc','text')})
    if calculator is not None:
        from .ic50 import concentration_comparison
        payload['tables']['workspace_ic50_calculator']=pd.DataFrame([concentration_comparison(*calculator)])
    payload['workspace_results']=records
    payload['tables']['workspace_section_coverage']=pd.DataFrame([
        {'Section':'Overview','Included':'Research workflow, local illustrative animation and interpretation limits'},
        {'Section':'Predict','Included':'All recorded session predictions and generated batches' if 'predict' in sections or 'batch' in sections or 'comparison' in sections else 'No prediction/batch was generated in this session; selected-pair evaluation is included separately'},
        {'Section':'Evaluation','Included':'Selected pair, full rankings, local explanations, all saved validation tables and figures, cursor-enabled charts'},
        {'Section':'Explainability','Included':'All recorded session SHAP results, plus selected-pair and saved global SHAP evidence' if 'explainability' in sections else 'Selected-pair and global SHAP evidence included; no separate Explainability-page result generated in this session'},
        {'Section':'Methods','Included':'Architecture, active training recipe, preprocessing, RWR, validation design and model metadata; deployment/run instructions excluded'},
    ])


def full_table_download(name,table):
    """All rows remain inside the HTML; avoid creating millions of DOM cells."""
    encoded=base64.b64encode(table.to_csv(index=False).encode('utf-8')).decode('ascii')
    preview=table.head(25)
    return (f'<details data-full-table="{escape(name,quote=True)}"><summary>{escape(name)} · {len(table):,} rows × {len(table.columns):,} columns</summary>'
        f'<a download="{escape(name)}.csv" href="data:text/csv;charset=utf-8;base64,{encoded}">Download complete embedded CSV · {len(table):,} rows</a>'
        f'<p>Preview: {len(preview):,} rows. Browse all rows below or download the complete embedded CSV.</p>'
        f'<button type="button" data-table-step="-1" disabled>Previous rows</button> '
        f'<button type="button" data-table-step="1" {"disabled" if len(table)<=25 else ""}>Next rows</button> '
        f'<output data-table-status>Rows 1–{len(preview):,} of {len(table):,}</output>'
        f'<div class="scroll">{preview.to_html(index=False,na_rep="Unavailable",escape=True)}</div></details>')



def markdown_science(text):
    """Escape source text before adding a small set of scientific-document styles."""
    lines=[]
    for line in text.splitlines():
        safe=escape(line)
        safe=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',safe)
        safe=re.sub(r'`([^`]+)`',r'<code>\1</code>',safe)
        if line.startswith('### '):lines.append('<h3>'+safe[4:]+'</h3>')
        elif line.startswith('## '):lines.append('<h3>'+safe[3:]+'</h3>')
        elif line.startswith('# '):lines.append('<h3>'+safe[2:]+'</h3>')
        elif line.strip():lines.append('<p>'+safe+'</p>')
    return '\n'.join(lines)

def section_content(payload):
    """Report scientific content from all five sections, excluding deployment steps."""
    metadata=payload.get('model_metadata',{})
    overview=(ROOT/'ui/molecule.html').read_text(encoding='utf-8')
    overview=overview.replace('/*CONFIG*/{}',json.dumps({'dark':False,'reduced':True,'compact':False}))
    parts=['<h2>Workspace coverage · all five sections</h2>',payload['tables']['workspace_section_coverage'].to_html(index=False,escape=True),
        '<h2>Overview · research workflow</h2><p>Explore drug chemistry and cancer cell-line pathway features, compare response estimates with measurements, inspect alternative research candidates and review model attributions. Features encode pathway enrichment, not one-drug-per-pathway assignments. No patient health records or clinical outcome model are supplied.</p>',
        '<p>The following local animation is illustrative scientific art, not a selected drug structure or experimental coordinate visualization.</p>',
        '<iframe title="Overview illustrative molecular animation" style="width:100%;height:750px;border:0" sandbox="allow-scripts" srcdoc="'+escape(overview,quote=True)+'"></iframe>',
        '<h2>Predict and Explainability · generated session results</h2><p>The complete tables below and evidence ZIP include every recorded result from this browser session. Analyses not run are marked as not generated; they are not invented. Each snapshot retains its own pair and model identity, which can differ from the Evaluation reference.</p>']
    for result in payload.get('workspace_results',[]):
        parts.append('<details><summary>'+escape(result['section'])+' · '+escape(result['generated_at_utc'])+'</summary><pre>'+escape(json.dumps(result['context'],indent=2,default=str))+'</pre><pre>'+escape(result['text'])+'</pre></details>')
    parts.append('<h3>Complete Overview section text</h3>')
    for step,title,description in OVERVIEW_CARDS:
        parts.append('<h3>'+escape(step+' · '+title)+'</h3><p>'+escape(description)+'</p>')
    for text in OVERVIEW_BLOCKS:parts.append(markdown_science(text))
    guide=ROOT/'SHAP_INTERPRETATION_GUIDE.md'
    if guide.is_file():
        parts.append('<h2>Explainability · complete interpretation guide</h2>'+markdown_science(guide.read_text(encoding='utf-8')))
    research_readme=ROOT/'research/validation-20261030/README.md'
    if research_readme.is_file():
        parts.append('<h2>Research validation findings</h2>'+markdown_science(research_readme.read_text()))
    parts.append('<h2>Complete Methods scientific content</h2>'+markdown_science(METHODS_SCIENCE))
    parts+=['<h2>Methods · scientific design</h2>',
        '<p>The model combines Morgan fingerprint chemistry, DG-Net drug-target pathway enrichment, and EXP/ssGSEA, MUT-Net and CNV-Net cell features. The FNN has hidden dimensions 1000, 800, 500 and 100, ELU activations and 0.1 dropout. Training-only feature imputation and standardization are saved with the model. The active recipe uses Xavier initialization, AdamW, MSE loss and early stopping on validation data. The evaluated checkpoint is served directly.</p>',
        '<p>Random Walk with Restart is used upstream by NetPEA for network-derived enrichment features. This app consumes the processed matrices; it does not claim to rerun the full upstream permutation normalization. Group SHAP is exact over five groups, while feature SHAP is approximate expected-gradient attribution. Positive SHAP raises −log10(IC50); these are model associations, not biological causal effects.</p>',
        '<p>Validation uses separate fitting, validation and pair-held-out rows and external CCLE evaluation. The historical holdout has been reused for model selection; drug/cell generalization and clinical efficacy are not established. Exploratory ROC/confusion labels use the training response median, not a clinical cutoff. Source: Tang and Gottlieb, Scientific Reports (2021), with upstream attribution preserved in the repository.</p>',
        '<h3>Active model metadata</h3><pre>'+escape(json.dumps(metadata,indent=2))+'</pre>']
    return '\n'.join(parts)


TABLE_PAGER_SCRIPT=r"""
<script>
(function(){
 function parseCSV(text){
  const rows=[];let row=[],field='',quoted=false;
  for(let index=0;index<text.length;index++){
   const char=text[index];
   if(char==='"'){
    if(quoted&&text[index+1]==='"'){field+='"';index++}else{quoted=!quoted}
   }else if(char===','&&!quoted){row.push(field);field=''}
   else if((char==='\n'||char==='\r')&&!quoted){
    if(char==='\r'&&text[index+1]==='\n')index++;
    row.push(field);rows.push(row);row=[];field='';
   }else{field+=char}
  }
  if(field.length||row.length){row.push(field);rows.push(row)}
  return rows;
 }
 document.addEventListener('click',event=>{
  const button=event.target.closest('[data-table-step]');if(!button)return;
  const details=button.closest('[data-full-table]');
  if(!details._csvRows){
   const encoded=details.querySelector('a[download]').href.split('base64,',2)[1];
   const bytes=Uint8Array.from(atob(encoded),char=>char.charCodeAt(0));
   details._csvRows=parseCSV(new TextDecoder('utf-8').decode(bytes)).slice(1);details._page=0;
  }
  const rows=details._csvRows,total=rows.length,last=Math.max(0,Math.ceil(total/25)-1);
  details._page=Math.max(0,Math.min(last,details._page+Number(button.dataset.tableStep)));
  const start=details._page*25,end=Math.min(total,start+25),body=details.querySelector('tbody');body.replaceChildren();
  for(const values of rows.slice(start,end)){
   const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr);
  }
  details.querySelector('[data-table-status]').textContent='Rows '+(total?start+1:0)+'–'+end+' of '+total;
  details.querySelector('[data-table-step="-1"]').disabled=details._page===0;
  details.querySelector('[data-table-step="1"]').disabled=details._page===last;
 });
})();
</script>"""
