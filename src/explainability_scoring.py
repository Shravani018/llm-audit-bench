import gc
import os
import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# gini = 0 -> attribution is perfectly uniform across all tokens (not explainable)
# gini = 1 -> all attribution is on one token (maximally concentrated)
def gini(values):
    """
    Computing Gini coefficient of an array of attribution values.
    Args:
        values: np.array of SHAP values per token
    Returns:
        float: Gini coefficient in [0, 1]
    """
    values = np.abs(values)
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values)) / (n * values.sum()) - (n + 1) / n)

def build_predict_fn(model, tokenizer, device):
    """
    Building a batched SHAP-compatible prediction function for a causal LM.
    Processes all texts in one forward pass, computing per-sample NLL
    while masking padding tokens from the loss.
    Args:
        model: loaded HuggingFace causal LM
        tokenizer: associated tokenizer
        device: 'cuda' or 'cpu'
    Returns:
        predict_fn: callable(list[str]) -> np.array of shape (n,)
    """
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    def predict_fn(texts):
        texts = list(texts)
        inputs = tokenizer(
            texts, return_tensors="pt",
            truncation=True, max_length=32,
            padding=True).to(device)
        if inputs["input_ids"].shape[1] < 2:
            return np.zeros(len(texts))
        with torch.no_grad():
            outputs = model(**inputs)
        # shifting for causal LM loss
        shift_logits = outputs.logits[:, :-1].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        shift_mask = inputs["attention_mask"][:, 1:].contiguous().float()
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(len(texts), -1)
        # masking padding tokens from per-sample mean loss
        per_sample_loss = (token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        return (-per_sample_loss).cpu().float().numpy()
    return predict_fn

def get_shap_values(explainer, sentence, max_evals=100):
    """
    Running SHAP on a single sentence and returning per-token attribution values.
    Capping max_evals to limit the number of predict_fn calls per sentence —
    default is 2*n_tokens+1 which explodes for longer inputs.
    Args:
        explainer: shap.Explainer instance
        sentence: input string
        max_evals: max SHAP perturbations per sentence (default 100)
    Returns:
        np.array of SHAP values (one per token), or None if SHAP fails
    """
    try:
        shap_values = explainer([sentence], max_evals=max_evals)
        values = shap_values.values[0]
        if values is None or len(values) == 0:
            return None
        return np.array(values, dtype=float)
    except Exception:
        return None

def compute_explainability(model, tokenizer, sentences, device, max_evals=100):
    """
    Computing explainability score for a model over a list of sentences.
    Args:
        model: loaded HuggingFace causal LM
        tokenizer: associated tokenizer
        sentences: list of input strings
        device: 'cuda' or 'cpu'
        max_evals: max SHAP perturbations per sentence passed to get_shap_values
    Returns:
        explainability_score: float in [0, 1]
        mean_gini: raw mean Gini across all sentences
        mean_top_tokens: average count of tokens holding >10% of attribution
    """
    predict_fn = build_predict_fn(model, tokenizer, device)
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker)
    gini_scores = []
    top_token_counts = []
    for sentence in tqdm(sentences, desc="computing SHAP"):
        values = get_shap_values(explainer, sentence, max_evals=max_evals)
        if values is None:
            continue
        g = gini(values)
        gini_scores.append(g)
        abs_vals = np.abs(values)
        total = abs_vals.sum()
        if total > 0:
            top_token_counts.append(int(np.sum(abs_vals / total > 0.10)))
    # deleting explainer/masker so predict_fn closure releases model reference
    # before the caller runs del model — prevents ghost VRAM retention
    del explainer, masker, predict_fn
    mean_gini = float(np.mean(gini_scores)) if gini_scores else 0.0
    mean_top_tokens = float(np.mean(top_token_counts)) if top_token_counts else 0.0
    explainability_score = round(mean_gini, 4)
    return explainability_score, round(mean_gini, 4), round(mean_top_tokens, 2)

def evaluate_explainability(model_id, sentences):
    """
    Evaluating explainability for a single model over the full sentence set.
    Fixing pad_token before model load so model.config stays in sync.
    Using float16 on CUDA for faster inference.
    Args:
        model_id: HuggingFace model identifier string
        sentences: list of input strings
    Returns:
        dict with model_id, explainability_score, mean_gini, mean_top_tokens, sentences_tested
    """
    print(f"\nEvaluating: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # fixing pad_token before model load so config.pad_token_id aligns
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # left-padding for causal LMs in batched mode
    tokenizer.padding_side = "left"
    # load config first and patch pad_token_id BEFORE model instantiation —
    # PhiConfig reads pad_token_id during __init__ (to set padding_idx),
    # so patching after from_pretrained is already too late.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_id)
    object.__setattr__(model_config, "pad_token_id", tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()
    try:
        explainability_score, mean_gini, mean_top_tokens = compute_explainability(
            model, tokenizer, sentences, device, max_evals=100
        )
    except Exception as e:
        print(f"  [ERROR] {model_id} failed: {e}")
        result = None
    else:
        print(f"  explainability: {explainability_score}  |  mean gini: {mean_gini}  |  avg top tokens: {mean_top_tokens}")
        result = {
            "model_id":             model_id,
            "explainability_score": explainability_score,
            "mean_gini":            mean_gini,
            "mean_top_tokens":      mean_top_tokens,
            "sentences_tested":     len(sentences),
        }
    finally:
        # releasing all references before cache clear so VRAM is fully freed
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result