---
slug: /
sidebar_position: 1
title: Welcome
---

# Welcome to Michelangelo

Michelangelo is an end-to-end ML platform for building, deploying, and managing machine learning models. Born at Uber — where it powers **25,000+ model trainings per month** and **~30 million predictions per second** — now open source.

## Why Michelangelo exists

Before Michelangelo, Uber's ML teams each built their own infrastructure: custom training pipelines, bespoke serving containers, no shared path to production. Every model was a one-off, and maintaining it was the team's problem alone. Michelangelo was created to solve that fragmentation — standardizing the full ML lifecycle so any team could ship production-quality ML without reinventing the stack.

Over eight years it evolved from tabular ML to deep learning to full LLMOps support, battle-tested across thousands of models and use cases. Open sourcing it extends that same mission: the platform is built on a modular, plug-and-play architecture so any engineering organization facing the same fragmentation problem can build on a foundation that has already been proven at scale. For the full story, see [History and Evolution](./about/history-and-evolution.md).

## Get started

### I'm evaluating Michelangelo

Understand what the platform does, how it compares to your current stack, and whether it fits your use case.

- **[Overview](./getting-started/overview.md)** — What Michelangelo is, how it works, and how familiar tools map to it
- **[Core Concepts](./getting-started/core-concepts-and-key-terms.md)** — Projects, workflows, tasks, and the key terms you'll encounter

### I want to build my first pipeline

Get a local environment running and build an end-to-end ML pipeline.

- **[Sandbox Setup](./getting-started/sandbox-setup.md)** — Set up a local Michelangelo cluster (~20 min)
- **[Getting Started with Pipelines](./user-guides/ml-pipelines/getting-started.md)** — Build your first pipeline from scratch (~30 min)
- **[Example Projects](https://github.com/michelangelo-ai/michelangelo/tree/main/python/examples)** — Boston Housing, BERT text classification, GPT fine-tuning, and more

### I'm deploying or operating the platform

Set up infrastructure, configure compute clusters, and deploy the UI.

- **[Operator Guides](./operator-guides/index.md)** — API framework, compute clusters, and serving infrastructure
- **[Building from Source](./contributing/building-michelangelo-ai-from-source.md)** — Compile and run the platform locally

### I want to contribute

- **[Documentation Guide](./contributing/documentation-guide.md)** — How to write and structure docs

## What Michelangelo is — and isn't

Understanding scope helps you decide if Michelangelo is the right tool.

**Michelangelo is:**
- An **ML lifecycle platform** — data prep, training, evaluation, deployment, and monitoring in one system
- An **orchestration framework** (Uniflow) for writing ML pipelines as Python code with `@task` and `@workflow` decorators
- A **model registry** for versioning, tracking, and managing trained models
- A **deployment system** for online inference (Triton) and batch predictions
- A **no-code UI** (MA Studio) for standard ML workflows without writing code

**Michelangelo is not:**
- A **notebook environment** — use Jupyter/Colab for exploration, then bring your code to Michelangelo for production
- A **data warehouse** — it connects to your existing data sources (S3, Snowflake, BigQuery, HDFS)
- A **general-purpose workflow engine** — it's purpose-built for ML, not arbitrary DAGs
- A **model monitoring SaaS** — monitoring is built in, but Michelangelo is self-hosted infrastructure, not a managed service
- A **replacement for your ML framework** — use PyTorch, TensorFlow, XGBoost, scikit-learn as you normally would

## Quick links

- [GitHub Repository](https://github.com/michelangelo-ai/michelangelo)
- [CLI Reference](./user-guides/cli.md)
- [ML Pipelines Overview](./user-guides/ml-pipelines/index.md)
