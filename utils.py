import re
from translate import translate
import torch

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def predict_bias_batch(tokenizer, model, sentences, max_length=128, device='cpu', bias_threshold=0.9, use_translation=True):
    results = []
    for item in sentences:
        if use_translation:
            en = item
            de = translate(en)
        else:
            en, de = item

        inputs = tokenizer(
            en,
            de,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        results.append((en, de, pred, confidence))
    return results
