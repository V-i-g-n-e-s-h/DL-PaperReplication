# DL-PaperReplication

# Citrus-Leaf Disease Classification with ACO-CNN

A Python package that replicates the **Ant Colony Optimised** CNN
pipeline for classifying four citrus‑leaf conditions: **black‑spot**, **canker**,
**greening** and **healthy**.

```bash
# 1.  Create and activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # on Linux/macOS
python -m venv .venv && .venv\Scripts\activate # on Windows

# 2.  Install package in editable (development) mode
pip install -e .

# 3.  Run 10‑fold cross‑validation (this takes a while!)
python -m citrus_aco_cnn
```