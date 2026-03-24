import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os
import tempfile
import subprocess
import matplotlib.pyplot as plt
from itertools import product
from Bio import SeqIO
from io import StringIO

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AMR Detective",
    page_icon="🧬",
    layout="wide"
)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/ann_model.pkl", "rb") as f:
        ann = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("data/resfams_mapping.json", "r") as f:
        resfams_map = json.load(f)
    return ann, le, resfams_map

ann, le, resfams_map = load_models()

# ── K-mer Function ────────────────────────────────────────────
def get_kmers(sequence, k=3):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    all_kmers = [''.join(p) for p in product(amino_acids, repeat=k)]
    kmer_index = {km: i for i, km in enumerate(all_kmers)}
    vector = np.zeros(len(all_kmers))
    seq = sequence.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_index:
            vector[kmer_index[kmer]] += 1
    total = vector.sum()
    if total > 0:
        vector = vector / total
    return vector

# ── HMMER Function ────────────────────────────────────────────
def run_hmmer(sequence):
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.fasta',
            delete=False, dir='data', prefix='query_'
        ) as tmp:
            tmp.write(f">query\n{sequence}\n")
            tmp_fasta = tmp.name

        tmp_output = tmp_fasta.replace('.fasta', '_out.txt')
        wsl_fasta = tmp_fasta.replace("\\", "/").replace("C:", "/mnt/c")
        wsl_output = tmp_output.replace("\\", "/").replace("C:", "/mnt/c")
        wsl_hmm = "/mnt/c/Users/Khatr/amr_tool/data/resfams/Resfams.hmm"

        cmd = [
            "wsl", "hmmscan",
            "--tblout", wsl_output,
            "--noali",
            "-E", "0.01",
            wsl_hmm,
            wsl_fasta
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        hits = []
        if os.path.exists(tmp_output):
            with open(tmp_output) as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        hits.append({
                            'Profile': parts[0],
                            'E-value': float(parts[4]),
                            'Score': float(parts[5])
                        })

        # Cleanup
        if os.path.exists(tmp_fasta):
            os.unlink(tmp_fasta)
        if os.path.exists(tmp_output):
            os.unlink(tmp_output)

        return sorted(hits, key=lambda x: x['E-value'])

    except Exception as e:
        return []

# ── ANN Prediction Function ───────────────────────────────────
def predict_ann(sequence):
    features = get_kmers(sequence, k=3).reshape(1, -1)
    predicted = le.inverse_transform(ann.predict(features))[0]
    probs = ann.predict_proba(features)[0]
    confidence = probs.max() * 100
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(le.classes_[i], round(probs[i]*100, 2)) for i in top3_idx]
    return predicted, confidence, top3

# ── UI ────────────────────────────────────────────────────────
st.title("🧬 AMR Detective")
st.markdown("### Antimicrobial Resistance Gene Detector")
st.markdown(
    "Upload or paste a **protein sequence** to detect AMR genes "
    "using Hidden Markov Models (HMMER/ResFams) and "
    "Artificial Neural Networks (k-mer + MLP)."
)

st.divider()

# ── Input Section ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Sequence")

    input_method = st.radio(
        "Choose input method:",
        ["Paste sequence", "Upload FASTA file"]
    )

    sequence = ""

    if input_method == "Paste sequence":
        raw_input = st.text_area(
            "Paste your protein sequence (FASTA or raw):",
            height=200,
            placeholder=">sequence_name\nMSIQHFRVALIPFFAAFCLP..."
        )

        if raw_input.strip():
            if raw_input.strip().startswith(">"):
                try:
                    rec = list(SeqIO.parse(StringIO(raw_input), "fasta"))
                    if rec:
                        sequence = str(rec[0].seq)
                        st.success(f"Parsed: {rec[0].id} ({len(sequence)} aa)")
                except:
                    st.error("Could not parse FASTA format")
            else:
                sequence = raw_input.strip().replace("\n", "").replace(" ", "")
                st.success(f"Sequence loaded ({len(sequence)} aa)")

    else:
        uploaded = st.file_uploader("Upload FASTA file", type=['fasta','fa','txt'])
        if uploaded:
            content = uploaded.read().decode('utf-8')
            try:
                rec = list(SeqIO.parse(StringIO(content), "fasta"))
                if rec:
                    sequence = str(rec[0].seq)
                    st.success(f"Loaded: {rec[0].id} ({len(sequence)} aa)")
            except:
                st.error("Could not parse uploaded file")

with col2:
    st.subheader("ℹ️ About This Tool")
    st.markdown("""
    **Two-layer detection system:**

    🔬 **HMM Layer (HMMER + ResFams)**
    - Scans sequence against 123 curated AMR HMM profiles
    - Identifies gene family with statistical confidence
    - E-value threshold: 0.01

    🤖 **ANN Layer (MLP Classifier)**
    - 3-mer frequency features (8000 dimensions)
    - Trained on 5948 CARD sequences
    - 11 drug resistance classes
    - 93.11% accuracy on held-out test set

    **Supported resistance classes:**
    beta-lactam, aminoglycoside, fluoroquinolone,
    tetracycline, macrolide, glycopeptide,
    phenicol, peptide, lincosamide,
    diaminopyrimidine, phosphonic acid
    """)

st.divider()

# ── Run Analysis ──────────────────────────────────────────────
if sequence:
    if len(sequence) < 50:
        st.warning("Sequence is very short (< 50 aa). Results may be unreliable.")

    if st.button("🔍 Run AMR Analysis", type="primary", use_container_width=True):

        with st.spinner("Running analysis..."):

            # Run both layers
            col_hmm, col_ann = st.columns([1, 1])

            # ── HMM Results ───────────────────────────────────
            with col_hmm:
                st.subheader("🔬 HMM Results")
                with st.spinner("Running HMMER..."):
                    hmm_hits = run_hmmer(sequence)

                if hmm_hits:
                    best = hmm_hits[0]
                    st.success(f"✅ AMR gene family detected!")

                    st.metric("Best Profile Match", best['Profile'])
                    st.metric("E-value", f"{best['E-value']:.2e}")
                    st.metric("Bitscore", f"{best['Score']:.1f}")

                    if len(hmm_hits) > 1:
                        st.markdown("**All significant hits:**")
                        df_hits = pd.DataFrame(hmm_hits[:8])
                        df_hits['E-value'] = df_hits['E-value'].apply(
                            lambda x: f"{x:.2e}"
                        )
                        st.dataframe(df_hits, use_container_width=True)
                else:
                    st.warning("⚠️ No ResFams profile matched")
                    st.markdown(
                        "This may be a novel AMR gene or outside "
                        "ResFams coverage. See ANN prediction."
                    )

            # ── ANN Results ───────────────────────────────────
            with col_ann:
                st.subheader("🤖 ANN Results")
                predicted, confidence, top3 = predict_ann(sequence)

                if confidence >= 80:
                    st.success(f"✅ Resistance class predicted!")
                elif confidence >= 50:
                    st.warning(f"⚠️ Low confidence prediction")
                else:
                    st.error(f"❌ Very low confidence — interpret carefully")

                st.metric("Predicted Class", predicted)
                st.metric("Confidence", f"{confidence:.1f}%")

                # Bar chart of top 3
                st.markdown("**Top 3 predictions:**")
                fig, ax = plt.subplots(figsize=(6, 3))
                classes = [t[0].replace(' antibiotic','') for t in top3]
                probs = [t[1] for t in top3]
                colors = ['#2ecc71' if i == 0 else '#95a5a6' for i in range(3)]
                bars = ax.barh(classes, probs, color=colors)
                ax.set_xlim(0, 100)
                ax.set_xlabel('Confidence (%)')
                ax.bar_label(bars, fmt='%.1f%%', padding=3)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── Summary ───────────────────────────────────────
            st.divider()
            st.subheader("📋 Summary")

            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                hmm_label = hmm_hits[0]['Profile'] if hmm_hits else "Not detected"
                st.metric("HMM Gene Family", hmm_label)
            with scol2:
                st.metric("ANN Drug Class", predicted)
            with scol3:
                st.metric("ANN Confidence", f"{confidence:.1f}%")

            # Agreement check
            if hmm_hits:
                hmm_drug = resfams_map.get(hmm_hits[0]['Profile'], None)
                if hmm_drug and hmm_drug == predicted:
                    st.success(
                        "✅ HMM and ANN predictions are in agreement — "
                        "high confidence result"
                    )
                elif hmm_drug:
                    st.info(
                        f"ℹ️ HMM suggests **{hmm_drug}** while ANN predicts "
                        f"**{predicted}** — consider both interpretations"
                    )

            # Download results
            st.divider()
            result_text = f"""AMR Detective Results
=====================
Sequence length: {len(sequence)} aa

HMM Analysis (HMMER + ResFams)
-------------------------------
{"Best match: " + hmm_hits[0]['Profile'] + " (E=" + f"{hmm_hits[0]['E-value']:.2e}" + ", score=" + str(hmm_hits[0]['Score']) + ")" if hmm_hits else "No profile matched"}
Total hits: {len(hmm_hits)}

ANN Analysis (MLP Classifier)
------------------------------
Predicted class: {predicted}
Confidence: {confidence:.1f}%

Top 3 predictions:
{chr(10).join([f"  {c}: {p}%" for c,p in top3])}
"""
            st.download_button(
                label="📥 Download Results",
                data=result_text,
                file_name="amr_results.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    st.info("👆 Please input a protein sequence above to begin analysis")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:grey'>"
    "AMR Detective | Built with HMMER + ResFams + CARD | "
    "MSc Bioinformatics Project"
    "</div>",
    unsafe_allow_html=True
)