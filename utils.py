import re
import streamlit as st
from typing import List, Tuple, Union
from translate import translate_batch
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model_and_tokenizer(model_dir: str, device: torch.device) -> Tuple[BertTokenizer, BertForSequenceClassification]:
    """
    Load and configure model and tokenizer
    Args:
        model_dir: Path to model directory
        device: Target device (cuda/cpu)
    Returns:
        Tuple of (tokenizer, model)
    """
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    return tokenizer, model

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation marks"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def predict_bias_in_translation_pairs(
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
    sentences: Union[List[str], List[Tuple[str, str]]],
    max_length: int = 128,
    device: torch.device = torch.device('cpu'),
    use_translation: bool = True
) -> List[Tuple[str, str, int, float]]:
    """
    Predict bias in sentence pairs
    Args:
        tokenizer: Text tokenizer
        model: Classification model
        sentences: List of English sentences or (English, German) pairs
        max_length: Maximum token length
        device: Target device
        use_translation: Whether to translate English to German
    Returns:
        List of (en_text, de_text, prediction, confidence) tuples
    """
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

def display_results(results: List[Tuple[str, str, int, float]], threshold: float) -> None:
    """
    Display prediction results with formatting
    Args:
        results: List of prediction tuples
        threshold: Confidence threshold for bias detection
    """
    for en, de, pred, conf in results:
        st.write(f"**English:** {en}")
        st.write(f"**German:** {de}")
        if pred == 1 and conf >= threshold:
            st.warning(f"⚠️ Bias detected (confidence: {conf:.2f})")
        else:
            st.success(f"✅ No bias detected (confidence: {conf:.2f})")
        st.markdown("---")