# ⚖️ llm-audit-bench
<p align="center">
  <img src="https://img.shields.io/badge/Transparency-Model_Cards-purple" />
  <img src="https://img.shields.io/badge/Fairness-CrowS--Pairs-blue" />
  <img src="https://img.shields.io/badge/Robustness-Perplexity_Shift-pink" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-navy" />
  <img src="https://img.shields.io/badge/Privacy-MIA_+_PII-orange" />
</p>

A modular pipeline that audits 5 small HuggingFace LLMs across transparency, fairness, robustness, explainability, and privacy.

## Pillars

| Pillar | Method |
|---|---|
| Transparency | Model card completeness scoring |
| Fairness | CrowS-Pairs stereotype bias test |
| Robustness | Perplexity shift under input perturbations |
| Explainability | SHAP token attribution |
| Privacy | MIA canary test + PII generation risk |
--

## Models

`gpt2` `distilgpt2` `facebook/opt-125m` `EleutherAI/gpt-neo-125m` `bigscience/bloom-560m'


## Dashboard
[View dashboard](https://shravani018.github.io/llm-audit-bench/)


## Notebooks

**01_extracting_metadata.ipynb**
- Fetches architecture and metadata via AutoConfig
- Saves to results/model_metadata.json

**02_transparency_score.ipynb**
- Scores completeness against 7 criteria: license, training data, limitations, intended use, evaluation results, carbon footprint, and card existence
- Each criterion is binary with a defined weight, producing a score between 0 and 1 per model

**03_fairness_score.ipynb**
- Measures stereotype bias across 9 demographic categories using CrowS-Pairs across 1508 sentence pairs
- Compares log-probabilities of stereotyped vs anti-stereotyped sentence pairs
- Produces a fairness score between 0 and 1 per model

**04_robustness_score.ipynb**
- Evaluates robustness by measuring perplexity shift under 3 input perturbations: typo, word deletion, and synonym substitution
- Uses 100 sentences from SST-2 and computes how much each model's output probability changes under slightly corrupted inputs
- Robustness score = 1 minus mean normalised perplexity shift across all three perturbation types

**05_explainability_score.ipynb**
- Measures token-level importance using SHAP attribution over 25 SST-2 sentences per model (nsamples=50, max_length=32)
- Explainability score derived from attribution concentration — a focused model assigns high importance to fewer, more meaningful tokens rather than spreading attribution uniformly

**06_privacy_score.ipynb**
- Evaluates privacy risk across two axes: MIA canary susceptibility and PII generation risk
- Privacy score = 1 minus normalised risk across both axes, where a higher score indicates a more privacy-preserving model

**07_aggregate_scores.ipynb**
- Aggregates all 5 pillar score JSONs into a single weighted trustworthiness index per model
- Weighted trust index: fairness 25%, robustness 25%, explainability 20%, transparency 15%, privacy 15%
  


## Limitations
**This was built as a learning exercise so the methodology has real constraints worth being upfront about.**

- The five models are all small and open-source, under 600M parameters. Nothing here generalises to instruction-tuned or larger models
- CrowS-Pairs has known quality issues and measures surface-level stereotype preference, not real-world harm
- Perplexity shift is a proxy for robustness, not a measure of adversarial or out-of-distribution resilience
- SHAP attribution concentration is one lens on interpretability and does not capture whether token importance aligns with human reasoning
- The canary memorisation test uses synthetic strings not seen during pre-training, so zero memorisation is expected and not a strong finding
 -  The trustworthiness index weights are manually set and not derived from any standard; different weights would change the rankings


## References

[Nangia et al. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. EMNLP.](https://aclanthology.org/2020.emnlp-main.154)

[Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.](https://arxiv.org/abs/1705.07874)

[Socher et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP.](https://aclanthology.org/D13-1170)

[Black et al. (2021). GPT-Neo. EleutherAI.](https://github.com/EleutherAI/gpt-neo)
