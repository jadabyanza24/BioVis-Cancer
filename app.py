import hashlib
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from preprocessing import ROOT, prepare_expression

st.set_page_config(page_title='LumaGene', page_icon=str(ROOT / 'assets' / 'biovis-logo.png'), layout='wide')

# A quiet, editorial workspace keeps attention on the sample and its output.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
.stApp { background: #fffafb; color: #392b33; font-family: 'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; color: #38232e; }
h1 { font-size: 2.9rem !important; line-height: 1.12 !important; }
h2 { font-size: 1.45rem !important; }
h3 { font-size: 1.1rem !important; }
p, label, [data-testid="stCaptionContainer"] { color: #66545e; }
[data-testid="stSidebar"] { background: #f9eef2; border-right: 1px solid #efdde4; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 1.15rem; }
[data-testid="stSidebar"] h2 { font-size: 1.15rem !important; }
.brand { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.25rem; color: #492936; letter-spacing: -.04em; }
.brand span { color: #bd4970; }
.hero-copy { font-size: 1.08rem; line-height: 1.6; max-width: 650px; color: #75616a; margin: .25rem 0 1.6rem; }
.workspace-label { color: #ac4669; font-weight: 700; font-size: .8rem; margin-bottom: .4rem; }
.workspace-intro { border-bottom: 1px solid #efdee5; padding-bottom: 1.2rem; margin-bottom: .35rem; }
.empty-state { background: #fff5f8; border: 1px solid #f3dce5; border-radius: 18px; padding: 2.2rem; min-height: 230px; display: flex; flex-direction: column; justify-content: center; }
.empty-icon { width: 44px; height: 44px; border-radius: 14px; background: #f4dbe5; color: #b73e68; display: grid; place-items: center; font-size: 1.3rem; margin-bottom: 1rem; }
.empty-state strong { font-family: 'Manrope', sans-serif; color: #4e303c; font-size: 1.1rem; }
.empty-state p { margin: .5rem 0 0; max-width: 330px; line-height: 1.55; }
.result-card { background: #f9e8ef; border: 1px solid #f1d4e0; border-radius: 16px; padding: 1.3rem 1.5rem; margin: .5rem 0 1rem; }
.result-card span { display: block; color: #835566; font-size: .85rem; }
.result-card strong { display: block; font-family: 'Manrope', sans-serif; font-size: 2rem; color: #a5325c; margin-top: .25rem; }
[data-testid="stBaseButton-primary"] { background: #bd426c; border: 0; border-radius: 9px; font-weight: 700; min-height: 2.9rem; }
[data-testid="stBaseButton-primary"] p { color: #fff; }
[data-testid="stBaseButton-primary"]:hover { background: #a9325b; }
[data-testid="stWidgetLabel"] p { color: #513d47; font-weight: 600; }
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: #efdee5; }
@media (max-width: 640px) { h1 { font-size: 2.2rem !important; } [data-testid="stMainBlockContainer"] { padding-top: 1rem; } }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    logo_column, brand_column = st.columns([1, 3.5], vertical_alignment='center')
    with logo_column:
        st.image(str(ROOT / 'assets' / 'biovis-logo.png'), width=48)
    with brand_column:
        st.markdown('<div class="brand">LumaGene</div>', unsafe_allow_html=True)
        st.caption('Gene expression explorer')
    st.divider()
    st.subheader('About')
    st.write('Explore how a gene expression model classifies LUAD stages and which features matter across the model.')
    st.subheader('Model workflow')
    st.write('Expression data is transformed and scaled, then filtered with ANOVA and RFE before XGBoost classification.')
    st.divider()
    st.caption('Research use only. Not for clinical decisions. Please do not upload sensitive data.')

st.markdown('<div class="workspace-label">Gene expression explorer</div>', unsafe_allow_html=True)
st.title('Explore a sample')
st.markdown('<p class="hero-copy">Choose an example or upload expression data to explore stage scores and global model features.</p>', unsafe_allow_html=True)

OUTPUT_MODEL = ROOT / 'outputs' / 'research_pipeline.joblib'
BUNDLED_MODEL = ROOT / 'artifacts' / 'research_pipeline.joblib'
MODEL_PATH = OUTPUT_MODEL if OUTPUT_MODEL.is_file() and OUTPUT_MODEL.with_name('evaluation.json').is_file() else BUNDLED_MODEL

@st.cache_resource
def load_pipeline(model_sha256):
    # The checksum matches the report but does not establish trust in a pickle.
    # Only load artifacts generated locally or obtained from a trusted source.
    return joblib.load(MODEL_PATH)

REPORT_PATH = MODEL_PATH.with_name('evaluation.json')
if not MODEL_PATH.is_file() or not REPORT_PATH.is_file():
    st.info('The evaluation model is unavailable. Run download_gdc.py and evaluate.py locally to create it.')
    st.stop()
try:
    report = json.loads(REPORT_PATH.read_text(encoding='utf-8'))
    model_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if model_sha256 != report.get('model_sha256'):
        raise ValueError('The model and evaluation report do not match. Recreate both files.')
    pipeline = load_pipeline(model_sha256)
    if 'variance' not in pipeline.named_steps or not pipeline.named_steps['model'].feature_importances_.any():
        raise ValueError('The model failed the feature-variation check. Run the evaluation again.')
except Exception as exc:
    st.error(f'Could not load the model. Check dependency versions and artifact provenance: {exc}')
    st.stop()

samples = sorted((ROOT / 'DataDummy').glob('*.tsv'))
input_column, output_column = st.columns([1, 1.45], gap='large')

with input_column:
    st.markdown('<div class="workspace-intro"><div class="workspace-label">01 / Input</div><h2>Choose a sample</h2><p>Select an example or bring your own expression file.</p></div>', unsafe_allow_html=True)
    mode = st.radio('Data source', ['Repository example', 'Upload a non-sensitive TSV'], horizontal=True)
    source = None
    if mode == 'Repository example':
        if samples:
            source = st.selectbox('Example file', samples, format_func=lambda path: f'Example {samples.index(path) + 1}')
            st.caption('Example files are included with the project.')
        else:
            st.info('No example files are available.')
    else:
        source = st.file_uploader('Upload expression data', type=['tsv', 'txt'], help='TSV must contain gene_id and fpkm_unstranded columns.')
        st.caption('TSV format · Required columns: gene_id, fpkm_unstranded')
    run = st.button('Run analysis', type='primary', disabled=source is None, use_container_width=True)

# Keep results across reruns, but never show an old result for a new source.
source_key = (mode, str(source) if mode == 'Repository example' else (source.file_id if source else None))
if st.session_state.get('source_key') != source_key:
    st.session_state.pop('analysis_result', None)
    st.session_state['source_key'] = source_key

if run and source is not None:
    try:
        frame = pd.read_csv(source, sep='\t', comment='#')
        aligned = prepare_expression(frame, pipeline.named_steps['scale'].feature_names_in_)
        prediction = pipeline.predict(aligned)[0]
        probabilities = pipeline.predict_proba(aligned)[0]
        classes = list(pipeline.classes_)
        if classes != [0, 1, 2, 3] or len(probabilities) != 4:
            raise ValueError('The model class mapping does not match the expected 0–3 labels.')
        feature_names = pipeline.named_steps['scale'].feature_names_in_[pipeline.named_steps['variance'].get_support()][pipeline.named_steps['anova'].get_support()][pipeline.named_steps['rfe'].get_support()]
        importance = pd.Series(pipeline.named_steps['model'].feature_importances_, index=feature_names).nlargest(10).sort_values()
        st.session_state['analysis_result'] = (int(prediction), probabilities, importance)
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        st.session_state.pop('analysis_result', None)
        with input_column:
            st.error(f'Could not analyze this file: {exc}')

with output_column:
    st.markdown('<div class="workspace-intro"><div class="workspace-label">02 / Results</div><h2>Model output</h2><p>Review the predicted class and scores for your sample.</p></div>', unsafe_allow_html=True)
    result = st.session_state.get('analysis_result')
    if result is None:
        st.markdown('<div class="empty-state"><div class="empty-icon">✳</div><strong>Your results will appear here</strong><p>Choose a sample and run the analysis to see its model output.</p></div>', unsafe_allow_html=True)
    else:
        prediction, probabilities, importance = result
        names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
        st.markdown(f'<div class="result-card"><span>Predicted class</span><strong>{names[prediction]}</strong></div>', unsafe_allow_html=True)
        st.subheader('Stage scores')
        st.caption('Model scores are not clinical probabilities.')
        st.bar_chart(pd.DataFrame({'Model score': probabilities}, index=names), color='#C34C75')
        st.divider()
        st.subheader('Global feature importance')
        st.caption('Top 10 genes across the model. Importance does not explain an individual sample or causation.')
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#fffafb')
        ax.set_facecolor('#fffafb')
        importance.plot.barh(ax=ax, color='#C34C75', width=.65)
        ax.set_xlabel('Model feature importance', color='#66545e')
        ax.tick_params(colors='#66545e', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
