import torch
from transformers import MarianMTModel, MarianTokenizer

model = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model)
model = MarianMTModel.from_pretrained(model)

def translate(text):
    input = tokenizer(text, return_tensors="pt", truncation=True)
    output = model.generate(**input)
    return tokenizer.decode(output[0], skip_special_tokens=True)

sentence="We need to prevent producers falling victim to the cyclical variation of the market."
print(translate(sentence)) 