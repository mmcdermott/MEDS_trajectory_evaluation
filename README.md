# Zero-shot ACES

This package contains utilities for converting autoregressive, generated trajectories into probabilistic
predictions for arbitrary ACES configuration files.

## 1. Install

```bash
pip install zeroshot_ACES
```

## 2. Run

```bash
ZSACES_predict ...
```

# Documentation

> [!IMPORTANT]
> This library only works with a subset of ACES configs; namely, those that have a tree-based set of
> dependencies between the end of the input window (the prediction time) and the end of the target window (the
> label window).
