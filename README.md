# ⚖️ llm-audit-bench

<p align="center">
  <img src="https://img.shields.io/badge/Transparency-Model_Cards-purple" />
  <img src="https://img.shields.io/badge/Fairness-CrowS--Pairs-blue" />
  <img src="https://img.shields.io/badge/Robustness-TextAttack-yellow" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-pink" />
  <img src="https://img.shields.io/badge/Privacy-MIA_+_PII-orange" />
</p>

A modular pipeline that audits 5 small HuggingFace LLMs across transparency, fairness, robustness, explainability, and privacy.

## Pillars
| Pillar | Method |
|---|---|
| Transparency | Model card completeness scoring |
| Fairness | CrowS-Pairs stereotype bias test |
| Robustness | TextAttack adversarial attacks |
| Explainability | SHAP token attribution |
| Privacy | MIA canary test + PII generation risk |

## Models
`gpt2` `distilgpt2` `facebook/opt-125m` `EleutherAI/gpt-neo-125m` `bigscience/bloom-560m`

## Status

**01_extracting_metadata.ipynb**
- Fetched architecture and metadata via AutoConfig.
- Saved to results/model_metadata.json

**02_transparency_score.ipynb**
- Scores completeness against 7 criteria: license, training data, limitations, intended use, evaluation results, carbon footprint, and card existence.
- Each criterion is binary (present / not present) with a defined weight
- Produces a transparency score between 0 and 1 per model.

**03_fairness_score.ipynb**
- Measures stereotype bias across demographic categories using CrowS-Pairs.
- Compares log-probabilities of stereotyped vs anti-stereotyped sentence pairs.
- Produces a fairness score between 0 and 1 per model.

**04_robustness_score.ipynb** *(in progress)*
- Evaluates robustness using TextAttack’s TextFooler adversarial attacks.
- Applies word-substitution attacks on a sentiment classification task.
- Reports Attack Success Rate (ASR) as the robustness metric.


