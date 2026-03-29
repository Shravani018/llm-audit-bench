import json
import os
import random
import torch
import numpy as np
import pandas as pd
import shap
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# gini = 0 -> attribution is perfectly uniform across all tokens (not explainable)
# gini = 1 -> all attribution is on one token (maximally concentrated)
def gini(values):
    """
    Computes Gini coefficient of an array of attribution values.
    Args:
        values: np.array of SHAP values per token
    Returns:
        float: Gini coefficient in [0, 1]
    """
    values = np.abs(values)
    values = np.sort(values)
    n      = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values)) / (n * values.sum()) - (n + 1) / n)

def build_predict_fn(model, tokenizer, device):
    """
    Builds a SHAP-compatible prediction function for a causal LM.
    Args:
        model: loaded HuggingFace causal LM
        tokenizer: associated tokenizer
        device: 'cuda' or 'cpu'
    Returns:
        predict_fn: callable(list[str]) -> np.array of shape (n,)
    """
    def predict_fn(texts):
        scores = []
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=32).to(device)
            if inputs["input_ids"].shape[1] < 2:
                scores.append(0.0)
                continue
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            scores.append(float(-outputs.loss.item()))
        return np.array(scores)
    return predict_fn
def get_shap_values(explainer, sentence):
    """
    Runs SHAP on a single sentence and returns per-token attribution values.
    Args:
        explainer: shap.Explainer instance
        sentence: input string
    Returns:
        np.array of SHAP values (one per token), or None if SHAP fails
    """
    try:
        shap_values = explainer([sentence])
        values      = shap_values.values[0]
        if values is None or len(values) == 0:
            return None
        return np.array(values, dtype=float)
    except Exception:
        return None
    
def compute_explainability(model, tokenizer, sentences, device):
    """
    Computes explainability score for a model over a list of sentences.
    Args:
        model: loaded HuggingFace causal LM
        tokenizer: associated tokenizer
        sentences: list of input strings
        device: 'cuda' or 'cpu'
    Returns:
        explainability_score: float in [0, 1]
        mean_gini: raw mean Gini across all sentences
        mean_top_tokens: average count of tokens holding >10% of attribution
    """
    predict_fn = build_predict_fn(model, tokenizer, device)
    masker    = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker)
    gini_scores      = []
    top_token_counts = []
    for sentence in tqdm(sentences, desc="computing SHAP"):
        values = get_shap_values(explainer, sentence)
        if values is None:
            continue
        g = gini(values)
        gini_scores.append(g)
        abs_vals = np.abs(values)
        total    = abs_vals.sum()
        if total > 0:
            top_count = int(np.sum(abs_vals / total > 0.10))
            top_token_counts.append(top_count)

    mean_gini       = float(np.mean(gini_scores))      if gini_scores      else 0.0
    mean_top_tokens = float(np.mean(top_token_counts)) if top_token_counts else 0.0
    explainability_score = round(mean_gini, 4)

    return explainability_score, round(mean_gini, 4), round(mean_top_tokens, 2)
def evaluate_explainability(model_id, sentences):
    """
    Evaluates explainability for a single model over the full sentence set.
    Args:
        model_id: HuggingFace model identifier string
        sentences: list of input strings
    Returns:
        dict with model_id, explainability_score, mean_gini, mean_top_tokens, sentences_tested
    """
    print(f"\nEvaluating: {model_id}")

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model     = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    explainability_score, mean_gini, mean_top_tokens = compute_explainability(
        model, tokenizer, sentences, device
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"  explainability: {explainability_score}  |  mean gini: {mean_gini}  |  avg top tokens: {mean_top_tokens}")

    return {
        "model_id":             model_id,
        "explainability_score": explainability_score,
        "mean_gini":            mean_gini,
        "mean_top_tokens":      mean_top_tokens,
        "sentences_tested":     len(sentences),
    }
     