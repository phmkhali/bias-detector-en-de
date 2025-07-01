import streamlit as st
from translate import translate

st.title("English to German Translation")

text = st.text_area("Enter English text here:")

if st.button("Translate"):
    if text:
        result = translate(text)
        st.write("German translation:")
        st.write(result)
    else:
        st.write("Please enter some text.")
