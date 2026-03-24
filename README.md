# 🧬 AMR Detective

A web-based tool for **Antimicrobial Resistance (AMR) gene detection** using a two-layer approach combining Hidden Markov Models and Artificial Neural Networks.

## Live Demo
> Run locally using instructions below

## Overview

AMR Detective takes a bacterial **protein sequence** as input and predicts:
- Which **AMR gene family** it belongs to (via HMMER + ResFams HMM profiles)
- Which **antibiotic resistance class** it confers (via trained MLP neural network)

## Two-Layer Architecture
```
Input Protein Sequence
        │
        ├──► HMM Layer (HMMER + ResFams)
        │    └── Gene family detection, E-value scoring
        │
        └──► ANN Layer (MLP Classifier)
             └── Drug class prediction, confidence scores
                      │
                      ▼
              Combined Result Report
```

## Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 93.11% |
| Training sequences | 5948 |
| Classes | 11 drug resistance classes |
| Features | 3-mer frequencies (8000 dimensions) |
| Architecture | MLP (256→128→64) |

## Resistance Classes Supported

- Beta-lactam antibiotic
- Aminoglycoside antibiotic
- Fluoroquinolone antibiotic
- Tetracycline antibiotic
- Macrolide antibiotic
- Glycopeptide antibiotic
- Phenicol antibiotic
- Peptide antibiotic
- Lincosamide antibiotic
- Diaminopyrimidine antibiotic
- Phosphonic acid antibiotic

## Known Limitations

- Classes with fewer than 100 training sequences (glycopeptide, lincosamide, phosphonic acid) show reduced F1 scores
- Lincosamide and macrolide share resistance mechanisms (erm methyltransferases) causing occasional misclassification
- ResFams covers 123 HMM profiles — novel or rare AMR genes may not be detected by HMM layer
- Tool is optimized for reference sequences, not fragmented metagenomic assemblies

## Installation
```bash
# Clone repository
git clone https://github.com/Khatri-ayush/AMR-Detective.git
cd AMR-Detective

# Create environment
conda create -n amr_tool python=3.10
conda activate amr_tool

# Install dependencies
pip install -r requirements.txt

# Install HMMER (Linux/WSL)
sudo apt install hmmer

# Download ResFams HMM profiles
mkdir -p data/resfams
wget -O data/resfams/Resfams.hmm.gz http://dantaslab.wustl.edu/resfams/Resfams.hmm.gz
gunzip data/resfams/Resfams.hmm.gz
hmmpress data/resfams/Resfams.hmm
```

## Usage
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## Data Sources

- **CARD** (Comprehensive Antibiotic Resistance Database) — training sequences
- **ResFams** — curated AMR HMM profiles
- **HMMER 3.4** — profile HMM search engine

## Project Structure
```
AMR-Detective/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── amr_exploration.ipynb   # Analysis notebook
├── models/
│   ├── ann_model.pkl       # Trained MLP classifier
│   └── label_encoder.pkl   # Class label encoder
└── data/
    ├── amr_labeled.csv         # Processed CARD dataset
    ├── resfams_mapping.json    # ResFams to drug class mapping
    ├── confusion_matrix_v2.png
    ├── training_curve_v2.png
    ├── class_distribution.png
    └── performance_vs_size.png
```

## Author

Ayush Khatri
MSc Bioinformatics, MDU Rohtak
