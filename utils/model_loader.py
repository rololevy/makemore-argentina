"""
Utility functions for the Streamlit app.
Handles model loading with caching.
"""

import os
import streamlit as st
import torch

# Add parent dir to sys.path so we can import makemore_ar
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from makemore_ar import load_checkpoint, MODEL_DESCRIPTIONS, MODEL_FRIENDLY_INFO


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Best to worst — the first available model will be selected by default
MODEL_PRIORITY = ['transformer', 'gru', 'rnn', 'mlp', 'bow', 'bigram']


def get_available_models(dataset_name: str) -> dict:
    """
    Scan models/ directory for trained checkpoints for a given dataset.
    Returns dict of {model_type: filepath}, ordered by quality (best first).
    """
    available = {}
    if not os.path.isdir(MODELS_DIR):
        return available
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith(f"{dataset_name}_") and fname.endswith('.pt'):
            model_type = fname.replace(f"{dataset_name}_", "").replace(".pt", "")
            available[model_type] = os.path.join(MODELS_DIR, fname)
    # Order by quality (best first)
    ordered = {k: available[k] for k in MODEL_PRIORITY if k in available}
    # Include any unknown model types at the end
    for k in available:
        if k not in ordered:
            ordered[k] = available[k]
    return ordered


@st.cache_resource
def load_model_cached(checkpoint_path: str):
    """Load a model checkpoint, cached by Streamlit."""
    model, dataset = load_checkpoint(checkpoint_path, device='cpu')
    return model, dataset
