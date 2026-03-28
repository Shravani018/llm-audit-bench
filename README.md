# ⚖️ llm-audit-framework
<p align="center">
  <img src="https://img.shields.io/badge/Transparency-Model_Cards-purple" />
  <img src="https://img.shields.io/badge/Fairness-CrowS--Pairs-blue" />
  <img src="https://img.shields.io/badge/Robustness-Perplexity_Shift-pink" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-navy" />
  <img src="https://img.shields.io/badge/Privacy-MIA_+_PII-orange" />
</p>

A modular pipeline that audits 5 small HuggingFace LLMs across transparency, fairness, robustness, explainability, and privacy.

---

## Models

`TinyLlama-1.1B-Chat-v1.0`, `phi-1_5`, `Qwen2-0.5B`, `SmolLM-360M`, `stablelm-2-1_6b`

| Pillar | Method | `0` → `1` |  Best Model |
|---|---|---|---|
| Transparency | Model card completeness | no docs → all criteria met |  |
| Fairness | CrowS-Pairs stereotype bias | fully biased → unbiased |  |
| Robustness | Perplexity shift under perturbations | noise-sensitive → fully stable | |
| Explainability | SHAP token attribution | diffuse → focused attribution |  |
| Privacy | MIA canary + PII generation risk | high risk → privacy-preserving |  |

---

## Dashboard
[View dashboard]()

---

## Notebooks

**01_extracting_metadata.ipynb**
- Fetches architecture and metadata via AutoConfig

**02_transparency_score.ipynb**
- Scores completeness against 7 criteria: license, training data, limitations, intended use, evaluation results, carbon footprint, and card existence
- Each criterion is binary with a defined weight.

**03_fairness_score.ipynb**
- Measures stereotype bias across 9 demographic categories using CrowS-Pairs across 1508 sentence pairs
- Compares log-probabilities of stereotyped vs anti-stereotyped sentence pairs

**04_robustness_score.ipynb**
- Evaluates robustness by measuring perplexity shift under 3 input perturbations: typo, word deletion, and synonym substitution
- Uses 100 sentences from SST-2 and computes how much each model's output probability changes under slightly corrupted inputs
- Robustness score = 1 minus mean normalised perplexity shift across all three perturbation types

**05_explainability_score.ipynb** *in progress*
- Measures token-level importance using SHAP attribution over 25 SST-2 sentences per model (nsamples=50, max_length=32)
- Explainability score derived from attribution concentration — a focused model assigns high importance to fewer, more meaningful tokens rather than spreading attribution uniformly

**06_privacy_score.ipynb**
- Evaluates privacy risk across two axes: MIA canary susceptibility and PII generation risk
- Privacy score = 1 minus normalised risk across both axes, where a higher score indicates a more privacy-preserving model

**07_aggregate_scores.ipynb**
- Aggregates all 5 pillar score JSONs into a single weighted trustworthiness index per model
- Weighted trust index: fairness 25%, robustness 25%, explainability 20%, transparency 15%, privacy 15%

---
  
## Results & Insights



---

## Limitations
**This was built as a learning exercise so the methodology has constraints**

- The five models are all small and open-source, under 600M parameters. Nothing here generalises to instruction-tuned or larger models
- CrowS-Pairs has known quality issues and measures surface-level stereotype preference, not real-world harm
- Transparency scores are based solely on model card completeness, which may not reflect the true openness of a model
- Perplexity shift is a proxy for robustness, not a measure of adversarial or out-of-distribution resilience
- SHAP attribution concentration is one lens on interpretability and does not capture whether token importance aligns with human reasoning
- The canary memorisation test uses synthetic strings not seen during pre-training, so zero memorisation is expected and not a strong finding
 -  The trustworthiness index weights are manually set and not derived from any standard but rather are based on understanding; different weights would change the rankings

---

## References

[Nangia et al. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. EMNLP.](https://aclanthology.org/2020.emnlp-main.154)

[Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.](https://arxiv.org/abs/1705.07874)

[Socher et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP.](https://aclanthology.org/D13-1170)

[Black et al. (2021). GPT-Neo. EleutherAI.](https://github.com/EleutherAI/gpt-neo)
