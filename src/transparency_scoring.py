from src import extracting_metadata as em
from huggingface_hub import HfApi, ModelCard
# Defining the criteris and weights(summing to 1.0)
criteria={
    "has_model_card":0.20,
    "license":0.15,
    "training_data":0.20,
    "limitations":0.15,
    "intended_use":0.10,
    "evaluation_results":0.10,
    "carbon_footprint":0.10
}
# Initializing API
api=HfApi()
# Fetching model card
def get_model_card(model_id):
    try:
        model_card=ModelCard.load(model_id)
        return model_card.content.lower()
    except Exception:
      return None
# Defining scoring mechanism for the model
def score_model_card(card_text,model_id):
  if card_text is None:
    return {k:False for k in criteria}
  checks={}
  # Does the card exist?
  checks["has_model_card"]=True #already extracted the text so it'll be True
  # Does the model have a license?
  checks['license']=em.get_license(model_id)
  # Is the training data described?
  checks["training_data"] = any(kw in card_text for kw in
        ["trained on", "training data", "dataset", "corpus", "pretraining", "fine-tuned on"])
  # Are the limitations mentioned?
  checks["limitations"] = any(kw in card_text for kw in
        ["limitation", "bias", "risk", "not suitable", "avoid", "failure", "caveat"])
  # Is the intended use described?
  checks["intended_use"] = any(kw in card_text for kw in
        ["intended use", "use case", "designed for", "primary use", "out-of-scope", "downstream"])
  # Are the evaluation results present?
  checks["evaluation_results"] = any(kw in card_text for kw in
        ["benchmark", "accuracy", "f1", "perplexity", "bleu", "rouge", "results", "performance", "score"])
  # Are the costs or carbon footprint mentioned?
  checks["carbon_footprint"] = any(kw in card_text for kw in
        ["carbon", "co2", "emissions", "compute", "gpu hours", "energy", "environmental"])
  return checks

# Calculating the score
def calc_score(checks):
  score=round(sum(criteria[k]*(1.0 if checks[k] else 0.0) for k in criteria),4)
  return score
# Scoring the model
def eval_transperancy(model_id):
  print(f"Evaluating Transperancy for:{model_id}")
  card_text=get_model_card(model_id)
  checks=score_model_card(card_text,model_id)
  score=calc_score(checks)
  stats={"model_id":model_id,"checks":checks,"score":score}
  return stats