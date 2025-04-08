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
- 📊 **Datasets**: mGeNTE corpus + building-bridges-gender-fair-german-mt dictionary + synthetic gender-swapped examples  
