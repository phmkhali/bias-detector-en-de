import streamlit as st
import torch
from utils import (
    load_model_and_tokenizer,
    split_sentences,
    predict_bias_in_translation_pairs,
    display_results
)
# Config
MODEL_DIR = "model_output"
MAX_LENGTH = 256
BIAS_CONF_THRESHOLD = 0.7

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    return load_model_and_tokenizer(MODEL_DIR, device)

def main():
    st.title("Gender Bias Detection in English-German Translations")
    
    tokenizer, model = load_resources()
    
    tab1, tab2 = st.tabs(["Translate", "Manual Input"])

    with tab1:
        text = st.text_area("Enter English text here:")
        if st.button("Translate and Check Bias"):
            if text.strip():
                with st.spinner("Splitting sentences..."):
                    sentences = split_sentences(text)
                
                with st.spinner("Translating and analyzing..."):
                    results = predict_bias_in_translation_pairs(
                        tokenizer,
                        model,
                        sentences,
                        max_length=MAX_LENGTH,
                        device=device,
                        use_translation=True
                    )
                
                display_results(results, BIAS_CONF_THRESHOLD)
            else:
                st.warning("Please enter some text")

    with tab2:
        manual_en = st.text_area("Enter English sentence:")
        manual_de = st.text_area("Enter German translation:")
        if st.button("Check Bias"):
            if manual_en.strip() and manual_de.strip():
                with st.spinner("Processing sentences..."):
                    en_sentences = split_sentences(manual_en)
                    de_sentences = split_sentences(manual_de)

                    if len(en_sentences) == len(de_sentences):
                        paired = list(zip(en_sentences, de_sentences))
                        
                        with st.spinner("Analyzing for bias..."):
                            results = predict_bias_in_translation_pairs(
                                tokenizer,
                                model,
                                paired,
                                max_length=MAX_LENGTH,
                                device=device,
                                use_translation=False
                            )
                        
                        st.success("Analysis complete!")
                        display_results(results, BIAS_CONF_THRESHOLD)
                    else:
                        st.error("The number of English and German sentences must match.")
            else:
                st.warning("Please enter both English and German text")

if __name__ == "__main__":
    main()