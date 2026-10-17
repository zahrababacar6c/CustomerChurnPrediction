"""Premium Streamlit interface for the IBM Telco churn prediction demo."""

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predict import load_model, predict_churn  # noqa: E402

st.set_page_config(
    page_title="Churn Intelligence | Decision Workspace",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#102E2D; --navy:#101C29; --sage:#90AE94; --ivory:#F7F5EF; }
      html, body, [class*="css"], [data-testid="stApp"] { font-family:'DM Sans',sans-serif; }
      [data-testid="stAppViewContainer"] { background:#F3F6FA; color:#101C29; }
      .block-container { max-width:1450px; padding-top:1rem; padding-bottom:3rem; }
      h1,h2,h3,h4 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
      [data-testid="stHeader"] { background:transparent; }
      [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:17px; }
      div[data-testid="stMetric"] { background:white; border:1px solid #E2E8E4; padding:1.1rem; border-radius:16px; }
      div[data-testid="stMetricValue"] { color:#073D76; font-weight:800; }
      .brandbar { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:1.4rem; }
      .brand { font-family:'Manrope',sans-serif; font-weight:800; color:#102E2D; letter-spacing:-.05em; font-size:1.1rem; }
      .brand span { color:#52806A; }
      .brandtag { color:#5B6D6C; font-size:.76rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
      .hero { border-radius:22px; padding:2.5rem 2.8rem; box-shadow:0 20px 38px rgba(7,43,98,.16); background:linear-gradient(112deg,#072B62 0%,#0753A5 58%,#063A74 100%); color:#F7F5EF; margin-bottom:1.6rem; }
      .hero .kicker { color:#B6D6C2; text-transform:uppercase; font-size:.75rem; letter-spacing:.18em; font-weight:800; }
      .hero h1 { color:#FFF; font-size:clamp(2rem,4.2vw,3.25rem); margin:.7rem 0; line-height:1.15; }
      .hero p { color:#DBE6E0; max-width:690px; line-height:1.65; margin-bottom:1.15rem; }
      .hero .pill { display:inline-block; border:1px solid #66877E; color:#E3F0E7; padding:.4rem .75rem; border-radius:30px; font-size:.75rem; margin-right:.45rem; margin-top:.4rem; }
      .sectionlabel { font-size:.73rem; letter-spacing:.14em; text-transform:uppercase; font-weight:800; color:#567B6B; margin-bottom:.35rem; }
      .sectionheading { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; color:#101C29; letter-spacing:-.04em; margin-bottom:.4rem; }
      .sectionnote { color:#60716E; font-size:.91rem; margin-bottom:1rem; }
      .micro { color:#61716C; font-size:.82rem; }
      .resultpanel { background:white; border:1px solid #E1E8E3; border-radius:18px; padding:1.3rem 1.5rem; }
      .resultpanel .eyebrow { color:#64816E; font-size:.73rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
      .resultpanel .bigscore { font-family:'Manrope',sans-serif; color:#102E2D; font-size:clamp(2.6rem,5vw,4rem); font-weight:800; letter-spacing:-.06em; line-height:1.25; }
      .resultpanel .sub { color:#60716E; font-size:.85rem; }
      .risk { border-radius:10px; display:inline-block; font-size:.8rem; font-weight:800; padding:.4rem .7rem; }
      .risk.high { background:#FFF0E8; color:#9A4529; }
      .risk.low { background:#E7F3EB; color:#25664A; }
      div.stButton > button[kind="primary"] { background:#0757BD; border:1px solid #0757BD; color:white; border-radius:12px; min-height:3.3rem; font-weight:800; }
      div.stButton > button[kind="primary"]:hover { background:#08478F; border-color:#08478F; }
      div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { border-radius:10px; }
      .navshell { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 20px; background:#FFFFFF; border:1px solid #E3E9F1; border-radius:15px; margin-bottom:16px; box-shadow:0 3px 16px rgba(18,49,86,.04); }
      .navlinks {display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
      .navitem {font-size:.84rem; font-weight:700; color:#64758B; padding:9px 13px; border-radius:9px;}
      .navitem.active {background:#E8F1FF; color:#0757BD;}
      .workspace-label {font-size:.76rem; letter-spacing:.14em; text-transform:uppercase; font-weight:800; color:#0757BD;}
      .hero-meta {display:flex; gap:12px; flex-wrap:wrap; margin-top:20px;}
      .hero-stat {border:1px solid rgba(255,255,255,.24); background:rgba(255,255,255,.08); border-radius:12px; padding:11px 16px; min-width:135px;}
      .hero-stat strong {display:block; font-size:1.08rem; color:white;}
      .hero-stat small {color:#CFE1F9; font-size:.74rem;}
      [data-testid="stVerticalBlockBorderWrapper"] {background:white; border-color:#DFE7F1!important; box-shadow:0 5px 16px rgba(23,53,89,.04);}
      div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {font-weight:600; color:#3E516A;}
      .resultpanel {border-color:#DCE7F4; box-shadow:0 7px 24px rgba(7,43,98,.06);}
      .resultpanel .bigscore {color:#0757BD;}
      .footer { border-top:1px solid #DEE6DF; padding-top:1.1rem; color:#63716D; font-size:.83rem; }
      @media(max-width:640px) { .hero { padding:1.5rem; } .brandtag { display:none; } .block-container { padding-top:.8rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """Load the trusted scikit-learn pipeline once per Streamlit process."""
    return load_model()


st.markdown(
    """<div class="navshell">
      <div class="brand">CHURN<span> / </span>INTELLIGENCE</div>
      <div class="navlinks"><span class="navitem active">Workspace</span>
      <span class="navitem">Customer profile</span><span class="navitem">Prediction insights</span></div>
      <div class="brandtag">APPLIED ML · PORTFOLIO</div>
    </div>""", unsafe_allow_html=True,
)
st.markdown(
    """<div class="hero">
      <div class="kicker">CUSTOMER ANALYTICS / DECISION WORKSPACE</div>
