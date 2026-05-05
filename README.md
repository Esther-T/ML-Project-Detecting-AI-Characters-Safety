## Project Overview
AI character platforms are a particularly popular domain where users can create and converse with AI-driven characters, including fictional characters, public figures, or entirely original creations. In this process, users can form emotional attachments and simulate relationships with AI characters for companionship, entertainment, and romance. Wei et al. (2025) have found that AI character platforms exhibited an average unsafe response rate of 65.1%, substantially higher than the 17.7% average rate of the general-purpose LLMs like ChatGPT. 

AI character platforms raise significant safety concerns due to the harmful content their characters may generate. This project investigates whether publicly available character metadata (i.e., descriptions, tags, scenarios, and NSFW designation) can predict a character's safety behavior without requiring direct interaction. We replicate the machine learning pipeline and data from Wei et al. (2025) using LLM-based annotation and classical classifiers, and extend it with an MLP trained on text embeddings. Classical classifiers achieve a best overall F1 of 0.63 (paper: 0.81), with the gap due to annotation quality. The MLP achieves a test F1 of 0.57, underperforming classical methods due to limited data scale. Both results confirm that character metadata carries meaningful predictive signals for safety classification.

## Team members
- Esther Tan
- Jialiang Yan

## File structure

```
├── data/
│   ├── annotated_checkpoint.json         # LLM annotation cache; append-only, supports resume
│   ├── annotated_with_nulls.json         # Raw LLM annotation output
│   ├── best_models.pkl                   # Saved traditional classifier models
│   ├── df.feather                        # Original data from Wei et al. (2025)
│   ├── encoded_with_nulls.feather        # Encoded features (partial annotations kept)
│   ├── ml_df.feather                     # Character-level pre-processed dataset
│   ├── ml_df_part1.feather               # Annotated batch 1 (characters 1–1030)
│   ├── nn_best_model.pkl                 # Best MLP model (single)
│   └── nn_best_models.pkl                # Best MLP models (all runs)
├── extra_results/                        # Additional experiment outputs
├── prompts/                              # Gemini annotation prompt templates
├── taxonomies/                           # YAML taxonomy files (6 dimensions)
├── 1_pre_processing.py                   # Data cleaning and label generation
├── 2_annotate.py                         # Gemini API annotation
├── 3_encoding.py                         # Feature encoding
├── 4_training.py                         # Traditional classifier training
├── 4_trainingNN.py                       # MLP training
└── README.md
```