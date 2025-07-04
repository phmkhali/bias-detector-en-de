import streamlit as st
from translate import translate
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load classifier and tokenizer once at start
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.eval()

st.title("English to German Translation with Gender Bias Detection")

text = st.text_area("Enter English text here:")

def predict_bias(english, german):
    text_pair = english + " [SEP] " + german
    inputs = tokenizer(text_pair, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
    return pred, confidence

if st.button("Translate"):
    if text:
        translation = translate(text)
        st.write("German translation:")
        st.write(translation)

        label, conf = predict_bias(text, translation)
        if label == 1 and conf >= 0.7:
            st.warning(f"The output might be gender biased. (Confidence: {conf:.2f})")
        else:
            st.success(f"No gender bias detected. (Confidence: {conf:.2f})")
    else:
        st.write("Please enter some text.")
