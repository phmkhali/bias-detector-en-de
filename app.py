import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import split_sentences, predict_bias_batch

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

st.title("Gender Bias Detection in English-German Translations")

tab1, tab2 = st.tabs(["Translate", "Manual Input"])

with tab1:
    text = st.text_area("Enter English text here:")
    if st.button("Translate and Check Bias"):
        if not text.strip():
            st.write("Please enter some text")
        else:
            st.write("### Results")
            sentences = split_sentences(text)
            results = predict_bias_batch(
                tokenizer,
                model,
                sentences,
                max_length=MAX_LENGTH,
                device=device,
                bias_threshold=BIAS_CONF_THRESHOLD,
                use_translation=True,
            )
            for sentence, translation, pred, conf in results:
                st.write(f"**english:** {sentence}")
                st.write(f"**german:** {translation}")
                if pred == 1 and conf >= BIAS_CONF_THRESHOLD:
                    st.warning(f"Bias detected confidence: {conf:.2f}")
                else:
                    st.success(f"No bias detected confidence: {conf:.2f}")
                st.markdown("---")

with tab2:
    manual_en = st.text_area("Enter English sentence:")
    manual_de = st.text_area("Enter German translation:")
    if st.button("Check Bias"):
        if not manual_en.strip() or not manual_de.strip():
            st.write("Please enter both English and German text")
        else:
            en_sentences = split_sentences(manual_en)
            de_sentences = split_sentences(manual_de)

            if len(en_sentences) != len(de_sentences):
                st.error("The number of English and German sentences must match.")
            else:
                paired_sentences = list(zip(en_sentences, de_sentences))
                results = predict_bias_batch(
                    tokenizer,
                    model,
                    paired_sentences,
                    max_length=MAX_LENGTH,
                    device=device,
                    bias_threshold=BIAS_CONF_THRESHOLD,
                    use_translation=False,
                )
                for sentence, translation, pred, conf in results:
                    st.write(f"**english:** {sentence}")
                    st.write(f"**german:** {translation}")
                    if pred == 1 and conf >= BIAS_CONF_THRESHOLD:
                        st.warning(f"Bias detected confidence: {conf:.2f}")
                    else:
                        st.success(f"No bias detected confidence: {conf:.2f}")
