import streamlit as st
import torch
from translate import translate
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("./model_output")
    model = BertForSequenceClassification.from_pretrained(
        "./model_output",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
        device_map=None
    )
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

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
        st.write("### Results")

        translation = translate(text)
        st.write(f"English: {text}")
        st.write(f"German: {translation}")

        label, conf = predict_bias(text, translation)
        if label == 1 and conf >= 0.7:
            st.warning(f"Bias detected (Confidence: {conf:.2f})")
        else:
            st.success(f"No bias detected (Confidence: {conf:.2f})")

        st.markdown("---")
    else:
        st.write("Please enter some text.")
