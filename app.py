import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown

if not os.path.exists("scaler_luad.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=FILE_ID",
        "scaler_luad.pkl",
        quiet=False
    )

# ==========================================
# 1. KONFIGURASI HALAMAN (Tampilan UI)
# ==========================================
st.set_page_config(
    page_title="BioVis-Cancer | LUAD Staging",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi warna visualisasi
sns.set_theme(style="whitegrid")
warna_stadium = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c'] # Ungu, Biru, Hijau, Merah

# ==========================================
# 2. MEMUAT KOMPONEN MODEL (Menggunakan Cache)
# ==========================================
@st.cache_resource
def load_pipeline():
    try:
        scaler = joblib.load('scaler_luad.pkl')
        selector_anova = joblib.load('selector_anova_luad.pkl')
        selector_rfe = joblib.load('selector_rfe_luad.pkl')
        model = joblib.load('model_stadium_luad.pkl')
        return scaler, selector_anova, selector_rfe, model
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file .pkl! Pastikan 4 file .pkl berada di folder yang sama dengan app.py. Error: {e}")
        st.stop()

scaler, selector_anova, selector_rfe, model = load_pipeline()
stage_names = ["Stadium I", "Stadium II", "Stadium III", "Stadium IV"]

# ==========================================
# 3. SIDEBAR (Informasi Aplikasi)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10619/10619850.png", width=100) # Ikon Paru-paru
    st.title("BioVis-Cancer")
    st.markdown("### Clinical Decision Support System")
    st.divider()
    st.markdown("""
    **Arsitektur Model:**
    * **Data:** RNA-Seq TCGA-LUAD
    * **Feature Selection:** ANOVA + RFE (150 Biomarker)
    * **Algoritma:** XGBoost (Optimized)
    """)
    st.divider()
    st.caption("Dikembangkan untuk analisis bioinformatika dan presisi medis.")

# ==========================================
# 4. HALAMAN UTAMA (Main Dashboard)
# ==========================================
st.title("Sistem Klasifikasi Stadium Kanker Paru-Paru (LUAD)")
st.markdown("""
Aplikasi ini memproses data rekam medis genomik (RNA-Sequencing) untuk mendeteksi tingkat keparahan (stadium) 
kanker paru-paru jenis *Lung Adenocarcinoma*. Silakan unggah file sekuensing pasien untuk memulai diagnosis.
""")

st.subheader("Input Data Pasien")
uploaded_file = st.file_uploader("Unggah file ekspresi gen (Format GDC STAR Counts .tsv):", type=['tsv', 'txt'])

if uploaded_file is not None:
    with st.spinner('Mengekstrak 150 biomarker esensial dan menjalankan analisis XGBoost...'):
        try:
            # --- FASE A: PREPROCESSING DATA INPUT ---
            df_temp = pd.read_csv(uploaded_file, sep='\t')
            df_temp = df_temp[df_temp['gene_id'].str.startswith('ENSG', na=False)]
            df_temp.set_index('gene_id', inplace=True)
            
            # Ambil kolom fpkm_unstranded dan jadikan 1 baris
            patient_data = df_temp['fpkm_unstranded'].to_frame().T
            
            # --- FASE B: ALIGNMENT & FILTERING (2-LAPIS) ---
            # 1. Sesuaikan urutan gen dengan data saat training
            training_genes = scaler.feature_names_in_ 
            patient_data = patient_data.reindex(columns=training_genes, fill_value=0.0)
            
            # 2. Transformasi Matematika (Log1p + Scaler)
            patient_log = np.log1p(patient_data)
            patient_scaled = scaler.transform(patient_log)
            
            # 3. Seleksi Fitur (ANOVA -> RFE)
            patient_anova = selector_anova.transform(patient_scaled)
            patient_selected = selector_rfe.transform(patient_anova)
            
            # --- FASE C: PREDIKSI AI ---
            prediction = model.predict(patient_selected)[0]
            probabilities = model.predict_proba(patient_selected)[0]
            predicted_stage = stage_names[prediction]
            
            # --- FASE D: MENAMPILKAN HASIL DIAGNOSIS ---
            st.divider()
            st.subheader("🔬 Hasil Analisis Komputasional")
            
            col1, col2 = st.columns([1.2, 2])
            
            # Kolom Kiri: Kesimpulan & Peringatan Medis
            with col1:
                st.markdown("### Kesimpulan Diagnosis")
                st.metric(label="Prediksi Kanker Pasien", value=predicted_stage)
                
                if prediction == 0:
                    st.success("**Kondisi:** Terditeksi pada fase awal (Stadium I).\n\n**Rekomendasi:** Intervensi bedah (reseksi) sangat disarankan. Peluang keberhasilan terapi tinggi.")
                elif prediction in [1, 2]:
                    st.warning(f"**Kondisi:** Terdeteksi pada fase penyebaran lokal ({predicted_stage}).\n\n**Rekomendasi:** Diperlukan observasi klinis ketat dan evaluasi untuk kemoterapi ajuvan.")
                else:
                    st.error("**Kondisi:** Terdeteksi pada fase metastasis lanjut (Stadium IV).\n\n**Rekomendasi:** Prioritas pada terapi sistemik (kemoterapi/targeted therapy) atau imunoterapi klinis.")
                    
            # Kolom Kanan: Visualisasi Tingkat Keyakinan (Probabilitas)
            with col2:
                st.markdown("### Tingkat Keyakinan Model (Probabilitas)")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(stage_names, probabilities * 100, color=warna_stadium, alpha=0.85)
                
                ax.set_ylabel("Probabilitas (%)", fontsize=10)
                ax.set_ylim(0, 105)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Tambahkan label persentase di atas setiap batang
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', 
                            ha='center', va='bottom', fontweight='bold', fontsize=11)
                
                st.pyplot(fig)
            
            # --- FASE E: CATATAN EDUKASI UNTUK AUDIENS ---
            st.info("""
            💡 **Interpretasi Biologis:** Jika probabilitas tersebar ke beberapa stadium (misal: 40% Stadium II dan 60% Stadium III), 
            ini mencerminkan profil mutasi RNA-Seq pasien yang sedang berada pada fase transisi (tumpang tindih genetik).
            """)
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data. Pastikan format file adalah TSV dari GDC Portal. Detail Error: {e}")