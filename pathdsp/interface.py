"""Presentation and catalogue search for PathDSP Lab."""
import json
from pathlib import Path
import streamlit as st

def apply_theme():
    brand, theme, motion = st.columns([5, 1.3, 1.5])
    brand.markdown("**🧬 PATHDSP LAB** · Research workspace")
    dark = theme.toggle("Dark mode", value=False, key="dark_mode")
    motion.toggle("Reduce motion", value=False, key="reduce_motion")
    st.markdown('''<style>
    .stApp{background:linear-gradient(145deg,#edf5f8,#fff 60%,#eaf7f3)}
    .block-container{max-width:1580px;padding:4rem 3rem 4rem}
    [data-testid="stHeader"]{background:transparent}
    .st-key-navigation [role="radiogroup"]{display:flex;gap:8px;flex-wrap:wrap;padding:8px;background:#b6d4df25;border:1px solid #aac7d640;border-radius:18px}
    .st-key-navigation [role="radiogroup"] label{padding:10px 18px;border-radius:12px;margin:0!important}
    .st-key-navigation [role="radiogroup"] label:has(input:checked){background:#137d7c;color:white!important}
    .st-key-navigation [role="radiogroup"] label:has(input:checked) p{color:white!important}
    .st-key-navigation [role="radiogroup"] label:focus-within{outline:2px solid #2b9f9b;outline-offset:2px}
    .st-key-navigation [role="radiogroup"] label:hover{box-shadow:inset 0 0 0 1px #65cbbb}
    .stApp:before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 85% 5%,#45b7ac13,transparent 50%);z-index:0}
    .workflow-card{transition:transform .25s,box-shadow .25s}
    .workflow-card:hover{transform:translateY(-5px);box-shadow:0 15px 40px #173b4c15}
    @media(max-width:700px){.block-container{padding:4rem 1rem 3rem}}
    @media(prefers-reduced-motion:reduce){.workflow-card{transition:none}.workflow-card:hover{transform:none}}
    h1,h2,h3{letter-spacing:-.035em;color:#102d40}
    div[data-testid="stMetric"]{background:white;border:1px solid #dbe9ee;border-radius:16px;padding:18px;box-shadow:0 5px 20px #173b4c08}
    .stButton>button[kind="primary"],.stFormSubmitButton>button{border-radius:12px;background:#117b79;color:white;border:0;padding:.65rem 1.2rem}
    [data-testid="stForm"]{background:#ffffffb8;border:1px solid #dbe9ee;border-radius:18px;padding:24px}
    .workflow-card{border:1px solid #dbe9ee;border-radius:18px;background:white;padding:22px;min-height:150px;margin:12px 0}
    .workflow-card small{color:#117b79;letter-spacing:2px}.workflow-card h3{font-size:22px;margin:8px 0}.workflow-card p{color:#536d7b;font-size:14px}
    </style>''', unsafe_allow_html=True)

    if st.session_state.get("reduce_motion"):
        st.markdown('<style>.workflow-card{transition:none!important}.workflow-card:hover{transform:none!important}</style>', unsafe_allow_html=True)
    if dark:
        st.markdown('''<style>
        :root{color-scheme:dark}
        .stApp{background:linear-gradient(145deg,#091925,#102b3b);color:#deedf5}
        h1,h2,h3,h4,p,label,[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"]{color:#deedf5!important}
        div[data-testid="stMetric"],[data-testid="stForm"],.workflow-card{background:#132e3f;border-color:#345264;color:#deedf5}
        .workflow-card p{color:#b8ceda!important}
        .workflow-card small{color:#7cf4d4}
        .notice{background:#342c21;color:#f8e7ce}
        [data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="input"] input,[data-testid="stTextInput"] input{background:#173648!important;color:#e6f3fa!important}
        [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"],[role="option"]{background:#173648!important;color:#e6f3fa!important}
        [data-testid="stExpander"]{background:#132e3f;border-color:#345264}
        .stButton>button,.stDownloadButton>button{background:#173e50;color:#e6f3fa;border-color:#477182}
        </style>''', unsafe_allow_html=True)


def molecular_intro(page="Overview"):
    """A responsive local illustration; no CDN, external structures or user HTML."""
    copy = {
        "Predict": ("From pathways to predictions.", "Find a drug–cell pair, run inference and compare the estimate with the measured response."),
        "Evaluation": ("Evidence before confidence.", "Separate development fit, held-out performance and cross-study transfer."),
        "Explainability": ("Understand the model’s reasoning.", "Explore how chemistry and pathway features contribute to a response estimate."),
        "Methods & deploy": ("Built on transparent science.", "Inspect the architecture, validation limits and reproducible research workflow."),
    }
    compact = page != "Overview"
    config = {"dark": st.session_state.get("dark_mode", False), "reduced": st.session_state.get("reduce_motion", False), "compact": compact,
              "scene": {"Predict": "molecule", "Evaluation": "pathway", "Explainability": "pathway", "Methods & deploy": "cell"}.get(page, "molecule")}
    if compact:
        config["title"], config["description"] = copy[page]
    html = (Path(__file__).resolve().parents[1] / 'ui/molecule.html').read_text(encoding='utf-8')
    html = html.replace('/*CONFIG*/{}', json.dumps(config))
    height = 225 if compact else 745
    if hasattr(st, 'iframe'):
        st.iframe(html, height=height, alt='Interactive scientific illustration with scene and motion controls')
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


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


def present_plot(fig):
    """Style this figure only, without changing shared Matplotlib defaults."""
    dark = st.session_state.get("dark_mode", False)
    background, foreground = ("#132e3f", "#deedf5") if dark else ("#ffffff", "#173b4c")
    fig.set_facecolor(background)
    for ax in fig.axes:
        ax.set_facecolor(background)
        ax.title.set_color(foreground)
        ax.xaxis.label.set_color(foreground)
        ax.yaxis.label.set_color(foreground)
        ax.tick_params(colors=foreground)
        for spine in ax.spines.values():
            spine.set_color("#527789" if dark else "#bed1db")
    st.pyplot(fig, clear_figure=True)
