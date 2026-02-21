# ⚖️ llm-audit-bench

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
- Fetched architecture and metadata via AutoConfig
- Saved to results/model_metadata.json

**02_transparency_score.ipynb** *(in progress)* 
- Scores each model's model card against completeness criteria.
