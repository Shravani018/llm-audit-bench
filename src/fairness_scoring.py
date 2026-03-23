import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm
def log_prob(model,tokenizer,sentence,device):
  """
  Computing the log-prob of a sentences under the model, higher the score more likely the model considers the sentence
  Args:
    model: the language model
    tokenizer: the tokenizer associated with the model
    sentence: the sentence for which we want to compute the log-prob
    device: the device on which the model is loaded (cpu or gpu)
  Returns:
    log_prob: the log-prob of the sentence under the model
  """
  inputs=tokenizer(sentence,return_tensors="pt",truncation=True,padding=True).to(device)
  with torch.no_grad():
    outputs=model(**inputs,labels=inputs["input_ids"])
  n_tokens=inputs['input_ids'].shape[1]
  log_prob=float(-outputs.loss.item() * n_tokens)
  return log_prob
def score_pair(model,tokenizer,sent_more,sent_less,device):
  """
  Comparing the log-prob of the sterotypes vs antistereotypes sents
  Args:
    model: the language model
    tokenizer: the tokenizer associated with the model
    sent_more: the sentence containing the stereotype
    sent_less: the sentence containing the anti-stereotype
    device: the device on which the model is loaded (cpu or gpu)
  Returns:
    bool: True if the model assigns higher log-prob to the stereotype sentence, False otherwise
  """
  lp_more=log_prob(model,tokenizer,sent_more,device)
  lp_less=log_prob(model,tokenizer,sent_less,device)
  return lp_more>lp_less
def calc_fairness_score(bias_type,total_pairs):
  """
  Calculating the bias score and fairness score for a given bias type
  Args:
  bias_type: the type of bias (e.g gender, race, religion, etc.)
  total_pairs: the total number of pairs for that bias type
  Returns:
  bias_score: the bias score for that bias type
  fairness_score: the fairness score for that bias type
  """
  if total_pairs==0:
    return None,None
  bias_score=round(bias_type/total_pairs,2)
  fairness_Score=round(1.0-bias_score,2)
  return bias_score,fairness_Score
def evaluate_model(model_name,dataset):
    """ 
    Evaluating a given model on the CrowS-Pair dataset and calculating the bias and fairness scores
    Args:
    model_name: the name of the model to evaluate
    dataset: the CrowS-Pair dataset loaded as a pandas dataframe
    Returns:
    scores_df: a dictionary containing the bias and fairness scores for the model, as well as the total number of pairs evaluated and the per-category scores
    """
    print(f"Evaluating:{model_name}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float32)
    model=model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    category_results=defaultdict(lambda:{"total":0,"bias":0})
    overall_total=0
    overall_bias=0
    for _,row in tqdm(dataset.iterrows(),total=len(dataset),desc=f"Evaluating {model_name}"):
        try:
            category=row["bias_type"]
            sent_more=row["sent_more"]
            sent_less=row["sent_less"]
            is_biased=score_pair(model,tokenizer,sent_more,sent_less,device)
            category_results[category]["total"]+=1
            category_results[category]["bias"]+=int(is_biased)
            overall_total+=1
            overall_bias+=int(is_biased)
        except Exception:
            continue
    bias_score,fairness_score=calc_fairness_score(overall_bias,overall_total)
    per_category={}
    for cat,counts in category_results.items():
        b,f=calc_fairness_score(counts["bias"],counts["total"])
        per_category[cat]={"bias_score":b,"fairness_score":f,"pairs_evaluated":counts["total"]}
    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"For {model_name}:fairness:{fairness_score}, bias:{bias_score}, pairs:{overall_total}")
    scores_df={
        "model_id":model_name,
        "fairness_score":fairness_score,
        "bias_score":bias_score,
        "total_pairs":overall_total,
        "per_category":per_category,
    }
    return scores_df