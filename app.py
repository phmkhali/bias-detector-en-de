import streamlit as st
import torch
from translate import translate
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./model_output" 
MAX_LENGTH = 128
BIAS_CONF_THRESHOLD = 0.9

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

st.title("English to German Translation with Gender Bias Detection")
text = st.text_area("Enter English text here:")

def predict_bias(english, german):
    inputs = tokenizer(
        english,
        german,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

if st.button("Translate"):
    if not text.strip():
        st.write("Please enter some text.")
    else:
        st.write("### Results")
        translation = translate(text)

        st.write(f"English: {text}")
        st.write(f"German: {translation}")

        label, conf = predict_bias(text, translation)

        if label == 1 and conf >= BIAS_CONF_THRESHOLD:
            st.warning(f"Bias detected (Confidence: {conf:.2f})")
        else:
            st.success(f"No bias detected (Confidence: {conf:.2f})")

        st.markdown("---")
