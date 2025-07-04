import streamlit as st
from translate import translate
import torch
from transformers import BertTokenizer, BertForSequenceClassification

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
