---
language: en
tags:
- self-supervised-learning
- consistency-training
- invariant-learning
- mathematical-reasoning
- temporal-reasoning
- causal-reasoning
license: mit
datasets:
- synthetic
---

# Consistency-Invariant Transformer (CIT)

A novel language model that learns through minimizing violation of theoretical invariants without any labeled data or human supervision.

## Model Description

The Consistency-Invariant Transformer (CIT) implements a groundbreaking approach to self-supervised learning. Instead of learning from labeled data or reinforcement learning from human feedback, CIT learns by minimizing its violation of known theoretical invariants across multiple domains:

- **Temporal Invariants**: Logical consistency in time relations (before/after/while)
- **Causal Invariants**: Asymmetry and transitivity in cause-effect relations
- **Mathematical Invariants**: Algebraic identities and equation consistency
- **Logical Invariants**: Propositional logic validity
- **Lexical Invariants**: Semantic consistency under paraphrasing
- **Factual Invariants**: Entity property consistency

## How It Works

The model optimizes the following objective:
