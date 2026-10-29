"""Presentation and catalogue search for PathDSP Lab."""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

def apply_theme():
    dark = st.sidebar.toggle("Dark mode", value=False, key="dark_mode")
    st.markdown('''<style>
    .stApp{background:linear-gradient(145deg,#edf5f8,#fff 60%,#eaf7f3)}
    .block-container{max-width:1320px;padding-top:2rem}
    [data-testid="stSidebar"]{background:#0b2233}
    [data-testid="stSidebar"] *{color:#d7edf4}
    [data-testid="stSidebar"] [role="radiogroup"] label{padding:10px;border-radius:10px}
    h1,h2,h3{letter-spacing:-.035em;color:#102d40}
    div[data-testid="stMetric"]{background:white;border:1px solid #dbe9ee;border-radius:16px;padding:18px;box-shadow:0 5px 20px #173b4c08}
    .stButton>button[kind="primary"],.stFormSubmitButton>button{border-radius:12px;background:#117b79;color:white;border:0;padding:.65rem 1.2rem}
    [data-testid="stForm"]{background:#ffffffb8;border:1px solid #dbe9ee;border-radius:18px;padding:24px}
    .workflow-card{border:1px solid #dbe9ee;border-radius:18px;background:white;padding:22px;min-height:150px;margin:12px 0}
    .workflow-card small{color:#117b79;letter-spacing:2px}.workflow-card h3{font-size:22px;margin:8px 0}.workflow-card p{color:#536d7b;font-size:14px}
    </style>''', unsafe_allow_html=True)

    if dark:
        st.markdown('''<style>
        :root{color-scheme:dark}
        .stApp{background:linear-gradient(145deg,#091925,#102b3b);color:#deedf5}
        h1,h2,h3,h4,p,label,[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"]{color:#deedf5!important}
        div[data-testid="stMetric"],[data-testid="stForm"],.workflow-card{background:#132e3f;border-color:#345264;color:#deedf5}
        .workflow-card p{color:#b8ceda!important}
        .notice{background:#342c21;color:#f8e7ce}
        [data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="input"] input,[data-testid="stTextInput"] input{background:#173648!important;color:#e6f3fa!important}
        [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"],[role="option"]{background:#173648!important;color:#e6f3fa!important}
        [data-testid="stExpander"]{background:#132e3f;border-color:#345264}
        .stButton>button,.stDownloadButton>button{background:#173e50;color:#e6f3fa;border-color:#477182}
        </style>''', unsafe_allow_html=True)


def molecular_intro():
    components.html((Path(__file__).resolve().parents[1] / 'ui/molecule.html').read_text(encoding='utf-8'), height=365)

def search_options(values, query):
    query = query.strip().casefold()
    return sorted((value for value in values if query in value.casefold()), key=str.casefold)

def searchable_pair(bundle, prefix):
    left, right = st.columns(2)
    drug_query = left.text_input('Search drug name', key=f'{prefix}_drug_search', placeholder='Type all or part of a drug name…')
    cell_query = right.text_input('Search cancer cell line', key=f'{prefix}_cell_search', placeholder='Type all or part of a cell-line name…')
    drugs = search_options(bundle.drug_ids, drug_query)
    cells = search_options(bundle.cell_ids, cell_query)
    left.caption(f'{len(drugs):,} matching drugs')
    right.caption(f'{len(cells):,} matching cell lines')
    if not drugs or not cells:
        st.info('No matching options. Try a shorter name or clear the search fields.')
    return drugs, cells
