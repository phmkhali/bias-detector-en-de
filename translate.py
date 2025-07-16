from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_batch(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
