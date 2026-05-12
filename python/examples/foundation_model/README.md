# Foundation Model — Training Example

End-to-end training pipeline for a multi-task sequence transformer that models
earner behavior. Built on the Michelangelo sequence SDK.

## What it does

1. **Feature prep** (Spark) — injects IDLE_DAY tokens, builds categorical vocabularies,
   normalizes numerical/geo features, creates fixed-length sequences, splits train/val.
2. **Training** (Ray + PyTorch Lightning) — trains `MultitaskSequenceLightning` with
   causal transformer backbone and per-task heads (next event type, churn).
3. **Upload** (Spark driver) — saves the best checkpoint to S3.

## Quick start

```bash
# 1. Edit config
vim examples/foundation_model/config.py   # set features, S3 bucket, hyperparams

# 2. Run locally (single machine, small data)
python examples/foundation_model/workflow.py

# 3. Submit to cluster
mactl submit examples/foundation_model/workflow.py
```

## Configuration

All tunable knobs live in `config.py`:

| Section | What to change |
|---|---|
| `PREP_CONFIG.categorical_features` | Which categorical columns to embed |
| `PREP_CONFIG.numerical_features` | Which numeric columns to normalize |
| `PREP_CONFIG.max_seq_length` | Sequence truncation length |
| `TRAIN_CONFIG.transformer_config` | Model size (d_model, n_heads, n_layers) |
| `TRAIN_CONFIG.task_config` | Which tasks to train, their weights and heads |
| `TRAIN_CONFIG.train_params` | Batch size, epochs, learning rate |
| `TRAIN_CONFIG.save_model_config.model_dir` | S3 or local checkpoint destination |

## File structure

```
examples/foundation_model/
├── config.py      # ALL concrete values (features, hyperparams, S3 paths)
└── workflow.py    # Uniflow DAG — wires tasks together, imports config

michelangelo/lib/foundation_model/
├── models/multitask_lightning.py        # MultitaskSequenceLightning
├── feature_prep/multitask_post_processor.py  # IDLE_DAY injection + sequential features
├── callbacks/{acceptance_rate,churn_auc}.py  # Validation metrics
└── tasks/{feature_prep,train,upload}_task.py # Uniflow task implementations

michelangelo/lib/sequence/
├── models/{mlp,encoders,task_heads,transformers/}  # Generic building blocks
├── lightning/base.py                               # BaseSequenceLightning
└── collate/sequence_collate.py                     # Numpy → tensor conversion
```

## Adding a new task head

1. Add a key to `TRAIN_CONFIG.task_config` in `config.py`.
2. Add the corresponding `pred_<name>_logits` entry to `_TASK_NAME_TO_PRED_OUTPUT`
   in `michelangelo/lib/foundation_model/model/multitask_lightning.py`.
3. Add the target column to the feature prep output in `tasks/feature_prep_task.py`.
