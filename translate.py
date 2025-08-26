from typing import List
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def translate_batch(sentences: List[str]) -> List[str]:
    """
    Translate a batch of English sentences into German.

    Args:
        sentences: List of English sentences to translate.

    Returns:
        List of German translations.
    """
    if not sentences:
        return []

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
