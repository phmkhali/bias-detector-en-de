# Bias-Marker-EN-DE  
**Detecting Gender Bias in English-German Translations using Natural Language Processing**  

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![NLP](https://img.shields.io/badge/NLP-BERT%2C%20OPUS--MT-orange)

A binary classification tool that flags gendered bias in English-to-German machine translations.

**Note:** This code repository is based on my bachelor thesis and is meant to be used alongside it for detailed explanations and interpretations.


## Features  
- 🚩 **Detects**:  
    Marks translations as gender biased when ambiguous subjects are translated with a specific gender or when gendered terms are incorrectly assigned. Other translations are considered neutral. **Examples:**  
  - Ambiguous subject defaulting to masculine: `doctor` → `Arzt`  
  - Stereotypical gender assignment: `nurse` → `Krankenschwester`  
  - Incorrect gender: `my mother is an engineer` → `meine Mutter ist ein Ingenieur`  


- 🔧 **Tech Stack**:  
  - Translation: OPUS-MT  
  - Web interface: Streamlit  
- 📊 **Dataset**: 
  A manually created dataset based on examples and datasets from previous studies in literature.


## Reproduction Guide

This guide explains how to set up and run the Streamlit demo app. You will create a Python virtual environment, install required packages, and run the app.  

**Note:** The pre-trained model is not included in the repository due to size restrictions. Download it separately via the provided Google Drive link. If you do not have the model, run `fine-tuning.ipynb` first to train and save the model before launching the demo.

## Installation Steps

1. Open a terminal (macOS/Linux) or PowerShell (Windows).

2. Clone the GitHub repository:

```bash
git clone https://github.com/phmkhali/bias-detector-en-de
cd bias-detector-en-de
```

3. Download the model

Manually download from Google Drive and place the `model_output` folder into the project directory: [Google Drive link](https://drive.google.com/drive/u/1/folders/11WMb0od_U_sQsUGD0t4DjQwcefI3r_kK)


4. Create and activate a Python virtual environment.

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

5. Install required packages

**macOS / Linux:**

```bash
pip3 install -r requirements.txt
```

**Windows:**

```bash
pip install -r requirements.txt
```

6. (Optional) If model_output is not downloaded, run fine-tuning.ipynb manually to generate the model

7. Run the Streamlit app

**macOS / Linux:**

```bash
python3 -m streamlit run app.py
```

**Windows:**

```bash
streamlit run app.py
```