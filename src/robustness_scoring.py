# Importing necessary libraries
import json
import os
import math
import random
import string
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet
warnings.filterwarnings("ignore")
random.seed(42)
#Loading 100 sentences from SST-2
sst2=load_dataset("sst2", split="validation")
sentences=[row["sentence"] for row in sst2.select(range(100))]

# finds synonyms
def get_synonym(word):
    synsets=wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            candidate=lemma.name().replace("_", " ")
            if candidate.lower()!=word.lower():
                return candidate
    return word
#Replaces the first word in the sentence with that synonym
def perturb_synonym(sentence):
    words=sentence.split()
    result=words[:]
    for i, word in enumerate(words):
        clean=word.lower().strip(string.punctuation)
        syn=get_synonym(clean)
        if syn!=clean:
            result[i]=syn
            break
    return " ".join(result)
# selects a random word and switches two chars to create a typo
def perturb_typo(sentence):
    words=sentence.split()
    if not words:
        return sentence
    idx=random.randint(0, len(words) - 1)
    word=list(words[idx])
    if len(word) > 2:
        swap=random.randint(0, len(word) - 2)
        word[swap], word[swap + 1] = word[swap + 1], word[swap]
    words[idx]="".join(word)
    return " ".join(words)
# removes a random word from the sentence
def perturb_delete(sentence):
    words=sentence.split()
    if len(words) <= 1:
        return sentence
    idx=random.randint(0, len(words) - 1)
    words.pop(idx)
    return " ".join(words)

perturbations={
    "typo":    perturb_typo,
    "deletion": perturb_delete,
    "synonym": perturb_synonym
}
     
#runs the sentence through the model and returns its perplexity
def get_perplexity(model, tokenizer, sentence, device):
    inputs = tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=128
    ).to(device)
    if inputs["input_ids"].shape[1] < 2:
        return None
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())
#Applying all 4 perturbations to each sentence, measures the perplexity shift before and after, and returns an overall robustness score plus per-perturbation breakdown.
def compute_robustness(model, tokenizer, sentences, device):
    per_type_shifts={k: [] for k in perturbations}
    all_shifts=[]
    for sentence in tqdm(sentences, desc="scoring sentences"):
        ppl_orig=get_perplexity(model, tokenizer, sentence, device)
        if ppl_orig is None or ppl_orig==0:
            continue
        for ptype, pfunc in perturbations.items():
            perturbed=pfunc(sentence)
            ppl_pert=get_perplexity(model, tokenizer, perturbed, device)
            if ppl_pert is None:
                continue
            shift=abs(ppl_pert - ppl_orig) / ppl_orig
            per_type_shifts[ptype].append(shift)
            if ptype != "shuffle":
                all_shifts.append(shift)
    mean_shift=float(np.mean(all_shifts)) if all_shifts else 1.0
    robustness_score=round(max(0.0, 1.0 - min(mean_shift, 1.0)), 4)
    per_type_scores={
        ptype: round(max(0.0, 1.0 - min(float(np.mean(shifts)), 1.0)), 4)
        if shifts else None
        for ptype, shifts in per_type_shifts.items()
        if ptype != "shuffle"
    }
    return robustness_score, mean_shift, per_type_scores
# Evaluating robustness for each model
def evaluate_robustness(model_id, sentences):
    print(f"\nEvaluating: {model_id}")
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model=model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    robustness_score, mean_shift, per_type = compute_robustness(
        model, tokenizer, sentences, device)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Robustness:{robustness_score}|mean shift: {round(mean_shift, 4)}")
    print(f"per type:{per_type}")
    return {
        "model_id":        model_id,
        "robustness_score": robustness_score,
        "mean_shift":       round(mean_shift, 4),
        "sentences_tested": len(sentences),
        "per_perturbation": per_type,
    }