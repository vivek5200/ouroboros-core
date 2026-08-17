# Ouroboros Core

Python/PyTorch diffusion engine for the Ouroboros v7.1 code refactoring system.

## Modules
- **Module 1 (Tokenizer)**: AST-aware tokenizer with Phantom Padding (L_max = 1024)
- **Module 3 (Attention)**: 1D RoPE + Additive AST Graph Bias attention mechanism
- **Module 4 (RL Reward)**: Fuzzy Proxy reward: R = 0.1*Parses + 0.3*TypeChecks + 0.6*PassesTests

## Setup
```bash
pip install -r requirements.txt
pytest tests/
```
