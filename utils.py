import re
from translate import translate_batch
import torch

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def predict_bias_batch(tokenizer, model, sentences, max_length=128, device='cpu', use_translation=True):
    results = []

    if use_translation:
        en_sentences = sentences
        de_sentences = translate_batch(en_sentences) 
        paired = zip(en_sentences, de_sentences)
    else:
        paired = sentences

    for en, de in paired:
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

