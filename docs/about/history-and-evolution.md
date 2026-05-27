---
sidebar_position: 1
sidebar_label: "History & Evolution"
---

# History and Evolution of Michelangelo

## The problem that started it

Before 2016, building and shipping a machine learning model at Uber meant starting from scratch every time. Applied scientists trained models in Jupyter notebooks. Engineers built custom serving containers for each project. There was no shared feature store, no standard deployment path, and no way for one team's infrastructure work to benefit any other. Every model that went to production was a one-off — and maintaining it was the team's own problem.

This wasn't a small inconvenience. Uber's business depends on ML: pricing, ETAs, maps, search, fraud detection, recommendations. Each of those domains was building its own version of the same infrastructure, in parallel, at increasing cost.

## What Michelangelo was built to do

Michelangelo was created in 2016 to solve the fragmentation problem at scale. The bet was that standardizing the end-to-end ML lifecycle — from feature engineering and training through deployment and monitoring — would let every team at Uber ship high-quality ML without reinventing infrastructure. A team working on Eats recommendations should be able to reuse the same platform primitives as a team working on fraud detection, with production-grade reliability built in.

That centralized model proved out. Over the following years, Michelangelo grew to serve thousands of models across hundreds of active projects, handling tens of millions of real-time predictions per second at peak.

## Eight years of evolution

What started as a platform for tabular ML and gradient-boosted trees expanded significantly as Uber's ML needs evolved:

- **Classical ML and XGBoost** (2016–2018) — The initial focus: standardizing feature pipelines, training jobs, and model serving for structured data problems.
- **Deep learning at scale** (2019–2021) — As neural networks became central to ranking, NLP, and computer vision use cases, Michelangelo added distributed training infrastructure, GPU support, and model architectures beyond XGBoost.
- **LLMOps and generative AI** (2022–present) — Foundation models, fine-tuning pipelines, retrieval-augmented generation, and agent-based workflows joined the platform as generative AI moved from experiment to production.

Each evolution built on the same core insight: platform investment compounds. The feature store, model registry, and deployment infrastructure that served classical ML became the foundation for deep learning — and is now the foundation for generative AI workflows.

## Why open source, and why now

The broader ML community has standardized on open, composable infrastructure: PyTorch for training, Ray for distributed compute, vLLM for inference. But there is still no dominant open-source end-to-end ML platform — no single place that organizes those building blocks into a coherent, production-ready system.

Open sourcing Michelangelo extends its original mission outward. The platform was built on a modular, plug-and-play architecture that cleanly separates platform primitives from Uber-specific internals. That design was intentional: it makes Michelangelo composable, so teams can adopt the full platform or integrate individual components into an existing stack.

This follows a pattern that has worked before. Google open sourced Kubernetes and it became the standard for container orchestration. Meta open sourced PyTorch and it became the standard language for ML research and production. Spotify open sourced Backstage and it became the default developer portal framework. In each case, the organization that open sourced a solved internal problem became the center of an ecosystem rather than a maintenance burden fighting ecosystem drift.

Michelangelo has already paid the hardest costs — platform design, operational hardening, battle-testing across thousands of models and use cases over eight years. The open source project packages those lessons so any engineering organization facing the same fragmentation problem can build on a foundation that has already been proven at scale.

## Further reading

- [From Predictive to Generative AI](https://www.uber.com/us/en/blog/from-predictive-to-generative-ai/) — Uber Engineering blog covering Michelangelo's architecture evolution in depth
- [From Monolith to Global Mesh: How Uber Standardized ML at Scale](https://thenewstack.io/uber-standardized-ml-scale/) — The New Stack on how Uber scaled ML across the organization
- [Michelangelo Deep Dive: Interview with Uber AI Platform PM](https://www.youtube.com/watch?v=x5cIMPmYAzw) — Interview covering history, GenAI use cases, tiering strategy, and open-source plans
- [Michelangelo Live Demo](https://www.youtube.com/watch?v=KJe8_FLMRx4) — Conference walkthrough of the full ML lifecycle, from project creation to deployment
- [Michelangelo Open Source Overview](https://www.youtube.com/watch?v=qXgCdkRNCYM) — Short conference talk on Michelangelo's history and open-source roadmap
