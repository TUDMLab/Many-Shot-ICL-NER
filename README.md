# Many-Shot ICL for Named Entity Recognition

Code and experiments for **Many-Shot In-Context Learning (ICL)** applied to
domain-specific / low-resource **Named Entity Recognition (NER)** across multiple
corpora, together with data-augmentation and fine-tuning baselines.

## Setup

```bash
pip install -r requirements.txt
```

### API keys

The LLM scripts call hosted APIs (OpenAI / DeepSeek / Google Gemini). **No keys are
committed to this repo** — placeholders such as `YOUR_API_KEY` / `YOUR_GEMINI_API_KEY`
appear where a key is needed. Provide your own, preferably via environment variables:

```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Repository layout

| Path | Description |
| --- | --- |
| `run_main.py`, `run_aug.py`, `run_ablation_main.py` | Main many-shot ICL + augmentation experiment drivers |
| `run_Gemini.py`, `run_deepseek.py`, `run_gpt4o.py`, `run_GliNER.py` | Model-specific runners |
| `run_Conll2003*.py`, `run_ner.py`, `run_entity_typing.py` | Dataset / task-specific runners |
| `fine_tune_bert*.py`, `run_slm.py` | Fine-tuning / small-LM baselines |
| `prompts.py`, `aug_prompt.py`, `templates.py`, `const.py` | Prompt templates and constants |
| `dataloader.py`, `utils.py`, `compute_metrics.py`, `eval_performance.py` | Data loading, utilities and evaluation |
| `datasets/` | Reference datasets (small) |
| `R2GRPO/` | RL (GRPO) experiments |
| `*.ipynb` | Exploratory data analysis, error analysis and visualization notebooks |

## Data & artifacts

Large model outputs, checkpoints, augmented corpora and unlabeled data are **not**
tracked in git (see `.gitignore`): `output/`, `ckpts/`, `ner_bert_checkpoints*/`,
`unlabeled/`. Regenerate them with the scripts above or obtain them separately.
