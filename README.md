# ⚖️ llm-audit-framework
<p align="center">
  <img src="https://img.shields.io/badge/Transparency-Model_Cards-purple" />
  <img src="https://img.shields.io/badge/Fairness-CrowS--Pairs-blue" />
  <img src="https://img.shields.io/badge/Robustness-Perplexity_Shift-pink" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-navy" />
  <img src="https://img.shields.io/badge/Privacy-MIA_+_PII-orange" />
</p>

A modular pipeline that audits 5 small HuggingFace LLMs across transparency, fairness, robustness, explainability, and privacy.

## Models

`gpt2` `distilgpt2` `facebook/opt-125m` `EleutherAI/gpt-neo-125m` `bigscience/bloom-560m`

| Pillar | Method | `0` → `1` |  Best Model |
|---|---|---|---|
| Transparency | Model card completeness | no docs → all criteria met | `distilgpt2` & `bloom-560m` with 1.0 |
| Fairness | CrowS-Pairs stereotype bias | fully biased → unbiased | `EleutherAI/gpt-neo-125m` with  0.46 |
| Robustness | Perplexity shift under perturbations | noise-sensitive → fully stable | `gpt2` with 0.46 |
| Explainability | SHAP token attribution | diffuse → focused attribution | `bigscience/bloom-560m` with 0.50 |
| Privacy | MIA canary + PII generation risk | high risk → privacy-preserving | `opt-125m` & `gpt-neo-125m` with 1.0 |

## Dashboard
[View dashboard](https://shravani018.github.io/llm-audit-framework/)


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

**05_explainability_score.ipynb**
- Measures token-level importance using SHAP attribution over 25 SST-2 sentences per model (nsamples=50, max_length=32)
- Explainability score derived from attribution concentration — a focused model assigns high importance to fewer, more meaningful tokens rather than spreading attribution uniformly

**06_privacy_score.ipynb**
- Evaluates privacy risk across two axes: MIA canary susceptibility and PII generation risk
- Privacy score = 1 minus normalised risk across both axes, where a higher score indicates a more privacy-preserving model

**07_aggregate_scores.ipynb**
- Aggregates all 5 pillar score JSONs into a single weighted trustworthiness index per model
- Weighted trust index: fairness 25%, robustness 25%, explainability 20%, transparency 15%, privacy 15%
  
## Results & Insights

- `distilgpt2` came out on top overall (0.61), ironically the smallest model in the set, with solid privacy, perfect transparency, and no critical failure modes. Robustness was where models separated most sharply
- `bloom-560m` scored a flat 0.0 due to extreme perplexity shift under typo perturbations (mean shift 4.67 vs ~0.54 for `gpt2`), which would be a hard blocker in any production setting.
- Fairness was weak across the board (0.42–0.46), with sexual orientation consistently the most biased category, suggesting the problem is training data rather than architecture. Privacy scores were high but should be interpreted cautiously, as zero memorisation across just 10 canaries is too small a sample to draw strong conclusions.
- Transparency scores are based solely on model card completeness, so a model with vague or outdated documentation can still score 1.0. If deploying one today, `distilgpt2` is the most defensible choice, while `facebook/opt-125m` and `bigscience/bloom-560m` both carry robustness risk worth flagging.

## Limitations
**This was built as a learning exercise so the methodology has real constraints worth being upfront about.**

- The five models are all small and open-source, under 600M parameters. Nothing here generalises to instruction-tuned or larger models
- CrowS-Pairs has known quality issues and measures surface-level stereotype preference, not real-world harm
- Transparency scores are based solely on model card completeness, which may not reflect the true openness of a model
- Perplexity shift is a proxy for robustness, not a measure of adversarial or out-of-distribution resilience
- SHAP attribution concentration is one lens on interpretability and does not capture whether token importance aligns with human reasoning
- The canary memorisation test uses synthetic strings not seen during pre-training, so zero memorisation is expected and not a strong finding
 -  The trustworthiness index weights are manually set and not derived from any standard; different weights would change the rankings


## References

[Nangia et al. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. EMNLP.](https://aclanthology.org/2020.emnlp-main.154)

[Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.](https://arxiv.org/abs/1705.07874)

[Socher et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP.](https://aclanthology.org/D13-1170)

[Black et al. (2021). GPT-Neo. EleutherAI.](https://github.com/EleutherAI/gpt-neo)
