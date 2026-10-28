import hashlib
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from preprocessing import ROOT, prepare_expression

st.set_page_config(page_title='BioVis-Cancer | Research Prototype', page_icon=str(ROOT / 'assets' / 'biovis-logo.png'), layout='wide')
logo_column, title_column = st.columns([1, 8], vertical_alignment='center')
with logo_column:
    st.image(str(ROOT / 'assets' / 'biovis-logo.png'), width=88)
with title_column:
    st.title('BioVis-Cancer')
st.subheader('Eksplorasi ekspresi gen dan stadium LUAD')
st.caption('Proyek akademik tim · TCGA-LUAD · RNA sequencing')
with st.sidebar:
    st.header('Tentang proyek')
    st.write('BioVis-Cancer mengeksplorasi klasifikasi stadium menggunakan data ekspresi gen dan menampilkan fitur penting model.')
    st.markdown('**Alur model**')
    st.write('Transformasi log, scaling, penyaringan gen konstan, seleksi ANOVA dan RFE, lalu XGBoost.')
    st.markdown('**Batasan**')
    st.write('Hasil belum divalidasi secara klinis dan tidak boleh digunakan untuk keputusan kesehatan.')
st.warning('Prototipe akademik untuk eksplorasi, bukan alat diagnosis atau panduan terapi. Validasi klinis dan kinerja pada data independen belum dibuktikan. Jangan unggah data pasien atau data sensitif.')
st.write('Pilih sampel untuk melihat keluaran model dan gambaran feature importance global. Hasil ini bersifat eksploratif.')
st.divider()

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
    st.info('Model evaluasi ulang belum tersedia. Jalankan download_gdc.py dan evaluate.py di lingkungan lokal untuk membuatnya.')
    st.stop()
try:
    report = json.loads(REPORT_PATH.read_text(encoding='utf-8'))
    model_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if model_sha256 != report.get('model_sha256'):
        raise ValueError('Model dan laporan evaluasi tidak cocok; buat ulang keduanya.')
    pipeline = load_pipeline(model_sha256)
    if 'variance' not in pipeline.named_steps or not pipeline.named_steps['model'].feature_importances_.any():
        raise ValueError('Model gagal pemeriksaan variasi fitur; jalankan evaluasi ulang.')
except Exception as exc:
    st.error(f'Gagal memuat model. Periksa versi dependensi dan asal artefak: {exc}')
    st.stop()

stage_iv = report.get('test_report', {}).get('3', {})
if stage_iv.get('support', 0) > 0:
    st.info(
        f"Dalam evaluasi eksploratif yang tersimpan, recall stadium IV adalah "
        f"{stage_iv['recall']:.0%} pada {int(stage_iv['support'])} kasus uji. "
        'Jumlah kasus kecil; hasil tidak membuktikan kemampuan klinis atau generalisasi.'
    )

st.subheader('Pilih sampel')
samples = sorted((ROOT / 'DataDummy').glob('*.tsv'))
mode = st.radio('Sumber data', ['Contoh repository', 'Unggah TSV non-sensitif'], horizontal=True)
source = None
if mode == 'Contoh repository':
    st.caption('File ini tampaknya berasal dari data ekspresi penelitian, bukan data sintetis murni. Asal persis dan kemungkinan tumpang tindih dengan data pelatihan belum diverifikasi; label nama file bukan evaluasi.')
    if samples:
        source = st.selectbox('File contoh', samples, format_func=lambda p: p.name)
    else:
        st.info('Tidak ada file contoh.')
else:
    source = st.file_uploader('File TSV dengan kolom gene_id dan fpkm_unstranded', type=['tsv', 'txt'])

if source is not None and st.button('Jalankan analisis', type='primary'):
    try:
        frame = pd.read_csv(source, sep='\t', comment='#')
        aligned = prepare_expression(frame, pipeline.named_steps['scale'].feature_names_in_)
        prediction = pipeline.predict(aligned)[0]
        probabilities = pipeline.predict_proba(aligned)[0]
        classes = list(pipeline.classes_)
        if classes != [0, 1, 2, 3] or len(probabilities) != 4:
            raise ValueError('Pemetaan kelas model tidak sesuai dengan asumsi 0–3; perlu verifikasi tim.')
        names = ['Stadium I', 'Stadium II', 'Stadium III', 'Stadium IV']
        st.divider()
        st.subheader('Hasil model untuk sampel')
        st.metric('Kelas prediksi', names[int(prediction)])
        st.caption('Skor berikut adalah keluaran model, bukan probabilitas klinis yang telah dikalibrasi atau ukuran akurasi.')
        st.bar_chart(pd.DataFrame({'Skor model': probabilities}, index=names), color='#C93F69')
        feature_names = pipeline.named_steps['scale'].feature_names_in_[pipeline.named_steps['variance'].get_support()][pipeline.named_steps['anova'].get_support()][pipeline.named_steps['rfe'].get_support()]
        importance = pd.Series(pipeline.named_steps['model'].feature_importances_, index=feature_names).nlargest(10).sort_values()
        st.subheader('Feature importance global model: 10 gen teratas')
        st.caption('Importance ini berlaku untuk model secara keseluruhan, bukan penjelasan untuk sampel ini; tidak menunjukkan sebab atau biomarker yang tervalidasi.')
        fig, ax = plt.subplots(figsize=(9, 4))
        importance.plot.barh(ax=ax, color='#C93F69')
        ax.set_xlabel('Feature importance model')
        st.pyplot(fig)
        plt.close(fig)
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        st.error(f'Data atau artefak tidak kompatibel: {exc}')
