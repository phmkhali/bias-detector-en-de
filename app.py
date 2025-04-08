import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

st.title("English-German Translation")

input_text = st.text_area("Enter English text:")

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if st.button("Translate"):
    if input_text.strip() == "":
        st.error("Please enter some text to translate.")
    else:
        translated_text = translate(input_text)
        st.write("### Translated Text")
        st.success(translated_text)
