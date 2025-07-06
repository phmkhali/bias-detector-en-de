import streamlit as st
import torch
import nltk
from nltk.tokenize import sent_tokenize
from translate import translate
from transformers import BertTokenizer, BertForSequenceClassification

# Download tokenizer for sentence splitting
nltk.download("punkt")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and fine-tuned model
tokenizer = BertTokenizer.from_pretrained("./model_output")
model = BertForSequenceClassification.from_pretrained("./model_output")
model.to(device)
model.eval()

st.title("English to German Translation with Gender Bias Detection")

text = st.text_area("Enter English text here:")

def predict_bias(english, german):
    text_pair = english + " [SEP] " + german
    inputs = tokenizer(
        text_pair,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
    return pred, confidence

if st.button("Translate"):
    if text.strip():
        sentences = sent_tokenize(text)
        st.write("### Results")

        for i, sentence in enumerate(sentences, 1):
            st.markdown(f"**Sentence {i}:**")
            st.write(f"English: {sentence}")

            translation = translate(sentence)
            st.write(f"German: {translation}")

            label, conf = predict_bias(sentence, translation)
            if label == 1 and conf >= 0.7:
                st.warning(f"Bias detected (Confidence: {conf:.2f})")
            else:
                st.success(f"No bias detected (Confidence: {conf:.2f})")

            st.markdown("---")
    else:
        st.write("Please enter some text.")
