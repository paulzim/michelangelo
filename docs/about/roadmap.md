---
sidebar_position: 2
sidebar_label: "Roadmap"
---

# Roadmap

Michelangelo is under active development. This page captures the current state of the platform and the direction we're headed — it is intentionally not a detailed spec or a delivery commitment. Things will shift as priorities evolve and the community gives feedback.

## Available Now

These capabilities are shipped and available in the current release.

**Project & Pipeline Management**
- Project creation and lifecycle management
- Pipeline authoring in YAML, Python, and ASL/Uniflow
- Revision management and versioning
- Overridable parameters via Terrablob URLs
- Auto-flip triggers on master branch merge

**Pipeline Execution**
- Pipeline run execution
- Trigger-based runs (cron, interval, and batch rerun)
- Backfill runs
- Batch rerun

**Distributed Training**
- Ray job launch and management
- Spark job launch on Kubernetes and Peloton
- Persistent Ray clusters via RayCluster CRD
- Federated multi-cluster dispatch and status sync

**Model Serving**
- Inference server creation
- Deployment rollout strategies
- Endpoint traffic splitting and shadow routing
- Training insights

**Infrastructure & Compute**
- Compute cluster registration
- Resource pool selection
- Storage management via Terrablob

**Automation & Self-Healing**
- Revision-gated state transitions
- Condition engine pattern
- Federated multi-cluster status sync

## In Progress

These features are actively being built and will land in upcoming releases.

**Project Management**
- GenAI project flavor support
- Team ownership via uOwn

**Pipeline Authoring**
- Draft-based authoring workflow
- Dev/Prod environment labels derived from git branch

**Model Deployment**
- Automatic rollback on alert firing
- Decommission workflow with no-traffic validation gate
- DeploymentEvent tracking

**Infrastructure**
- Notebook (Jupyter) sessions
- Docker image builds via ImageBuild CRD

**Observability**
- Dashboard management
- Experiment reports

## On the Radar

These are areas we expect Michelangelo to grow into. None of them are committed features yet — they signal direction, not a fixed schedule.

**Generative AI & LLM**
- GenAI service deployment: first-class support for deploying and managing LLM-backed services within the Michelangelo lifecycle.

**Evaluation & Experimentation**
- A structured evaluation framework for comparing training runs, tracking metrics across experiments, and producing shareable experiment reports.

**Alerting & Monitoring**
- Full uMonitor integration for decommission gating and alert-driven automation across the deployment lifecycle.

---

The best way to influence what comes next is to open a GitHub issue or discussion with your use case. We treat this page as a living document and update it as concrete designs emerge.
