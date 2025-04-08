from transformers import MarianMTModel, MarianTokenizer

model = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model)
model = MarianMTModel.from_pretrained(model)

def translate(text):
    input = tokenizer(text, return_tensors="pt", truncation=True)
    output = model.generate(**input)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(translate("The surgeon is hard-working.")) 