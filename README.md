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

## Models
`gpt2` `distilgpt2` `facebook/opt-125m` `EleutherAI/gpt-neo-125m` `bigscience/bloom-560m`

## Status

**01_extracting_metadata.ipynb**
- Fetched architecture and metadata via AutoConfig
- Saved to results/model_metadata.json

**02_transparency_score.ipynb**
- Scored completeness against 7 criteria: license, training data, limitations, intended use, evaluation results, carbon footprint, and card existence
- Each criterion is binary with a defined weight, producing a score between 0 and 1 per model

**03_fairness_score.ipynb**
- Measures stereotype bias across 9 demographic categories using CrowS-Pairs across 1508 sentence pairs
- Compares log-probabilities of stereotyped vs anti-stereotyped sentence pairs
- Produces a fairness score between 0 and 1 per model

**04_robustness_score.ipynb**
- Evaluates robustness by measuring perplexity shift under 3 input perturbations: typo, word deletion, and synonym substitution
- Uses 100 sentences from SST-2 and computes how much each model's output probability changes under slightly corrupted inputs
- Robustness score = 1 - mean normalised perplexity shift across all three perturbation types

**05_explainability_score.ipynb** *(in progress)*
- Measures token-level importance using SHAP attribution over 100 SST-2 sentences per model
- Explainability score derived from attribution concentration, a focused model assigns high importance to fewer, more meaningful tokens rather than spreading attribution uniformly
