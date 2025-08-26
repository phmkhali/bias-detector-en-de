import streamlit as st
import torch
from utils import (
    load_model_and_tokenizer,
    split_sentences,
    predict_bias_in_translation_pairs,
    display_results
)

MODEL_DIR = "model_output"
MAX_LENGTH = 256
BIAS_CONF_THRESHOLD = 0.5  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    """Load the fine-tuned model and tokenizer."""
    return load_model_and_tokenizer(MODEL_DIR, device)

def process_text_analysis(tokenizer, model, text_pairs, use_translation=True):
    """
    Process input text for bias detection.

    Args:
        tokenizer: Tokenizer object.
        model: Bias detection model.
        text_pairs: Text input. Either a string (for translation) or a tuple of (en_text, de_text).
        use_translation: Whether to perform translation or manual input check.

    Returns:
        List of results with bias predictions and confidence scores, or None if an error occurs.
    """
    try:
        with st.spinner("Splitting sentences..." if use_translation else "Processing sentences..."):
            if use_translation:
                sentences = split_sentences(text_pairs)
                input_data = sentences
            else:
                en_sentences = split_sentences(text_pairs[0])
                de_sentences = split_sentences(text_pairs[1])
                if len(en_sentences) != len(de_sentences):
                    st.error("The number of English and German sentences must match.")
                    return None
                input_data = list(zip(en_sentences, de_sentences))

        with st.spinner("Translating and analyzing..." if use_translation else "Analyzing for bias..."):
            return predict_bias_in_translation_pairs(
                tokenizer,
                model,
                input_data,
                max_length=MAX_LENGTH,
                device=device,
                use_translation=use_translation
            )
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None

def show_disclaimer():
    """Display app disclaimer about bias detection limitations."""
    st.markdown(
        """
        **Disclaimer: Limitations of the Bias Detector**

        - Use of German gender-neutral forms (GFL) is sometimes wrongly classified as biased.
        - Male defaults in political and government roles may not always be flagged as biased.  
        - Semantically gendered words are sometimes wrongly classified as biased.  
        - Capitalization and punctuation can affect classification results.  
        - Sentences with multiple subjects may lower classification accuracy.  
        """,
        unsafe_allow_html=True
    )

def main():
    """Streamlit app entry point."""
    st.title("Gender Bias Detection in English-German Translations")
    
    tokenizer, model = load_resources()
    
    tab1, tab2 = st.tabs(["Translate", "Manual Input"])

    with tab1:
        text = st.text_area("Enter English text here:")
        if st.button("Translate and Check Bias"):
            if text.strip():
                results = process_text_analysis(tokenizer, model, text, use_translation=True)
                if results:
                    display_results(results, BIAS_CONF_THRESHOLD)
            else:
                st.warning("Please enter some text")

    with tab2:
        manual_en = st.text_area("Enter English sentence:")
        manual_de = st.text_area("Enter German translation:")
        if st.button("Check Bias"):
            if manual_en.strip() and manual_de.strip():
                results = process_text_analysis(
                    tokenizer, 
                    model, 
                    (manual_en, manual_de), 
                    use_translation=False
                )
                if results:
                    display_results(results, BIAS_CONF_THRESHOLD)
            else:
                st.warning("Please enter both English and German text")

    show_disclaimer()

if __name__ == "__main__":
    main()
