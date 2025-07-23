# Bias-Marker-EN-DE  
**Detecting Gender Bias in English-German Translations with NLP**  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/NLP-BERT%2C%20OPUS--MT-orange)

A binary classification tool that flags gendered bias in English-to-German machine translations.

## Features  
- 🚩 **Detects**:  
  - Added pronouns (e.g., "teacher" → "Lehrer" [masculine default])  
  - Incorrect noun gendering (e.g., "they" → "er/sie")  
- 🔧 **Tech Stack**:  
  - Translation: OPUS-MT  
  - Web interface: Streamlit  
- 📊 **Datasets**: 

## How to install

1. Clone the repository  
2. Make sure you have Python and pip installed  
3. Install the required packages:

   ```bash
   pip install -r requirements.txt

## How to run

1. Build the model:

   - Open `fine-tuning.ipynb`
   - Run all cells to train and save the model

2. Start the demo app:

   ```bash
   streamlit run app.py
