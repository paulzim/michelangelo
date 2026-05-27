---
sidebar_position: 1
sidebar_label: "RFCs"
---

# Michelangelo RFCs

Requests for Comments (RFCs) are the primary venue for community-visible design discussions. An RFC documents the problem, the proposed architecture, alternatives considered, and open questions — before implementation begins.

RFCs are maintained in the **[michelangelo-ai/enhancements](https://github.com/michelangelo-ai/enhancements)** repository on GitHub.

## RFC lifecycle

```
Idea → Draft PR → Community Review → Accepted / Withdrawn → Implementation
```

| Stage | What it means |
|---|---|
| **Draft** | RFC PR is open; design is in progress |
| **In Review** | RFC is complete and under active community review |
| **Accepted** | PR is merged; implementation PRs link back to the RFC |
| **Withdrawn** | PR is closed with a summary of why |

## Current RFCs

| RFC | Title | Status | Created |
|-----|-------|--------|---------|
| [20260427](https://github.com/michelangelo-ai/enhancements/blob/main/rfcs/20260427-michelangelo-helmchart/20260427-michelangelo-helmchart.md) | Michelangelo Control Plane Helm Chart | Accepted | 2026-04-27 |

## Writing an RFC

1. Copy the [RFC template](https://github.com/michelangelo-ai/enhancements/blob/main/rfcs/20260101-template.md) to `rfcs/YYYYMMDD-<short-name>.md` in the enhancements repo
2. Fill in each section — problem statement, motivation, goals, architecture, APIs, alternatives, open questions, and rollout strategy
3. Open a draft PR in [michelangelo-ai/enhancements](https://github.com/michelangelo-ai/enhancements)
4. Use GitHub PR comments for architecture discussion; link the PR in GitHub Discussions for broader input
5. Once accepted, link implementation PRs back to the RFC

Keep RFCs focused on architecture. Full implementation details and large code blocks belong in the implementation PRs.

For early-stage ideas that aren't ready for a formal RFC, open a [GitHub Issue](https://github.com/michelangelo-ai/michelangelo/issues) first to validate the direction.
