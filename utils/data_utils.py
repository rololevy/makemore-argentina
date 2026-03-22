"""
Data utilities for the Streamlit app.
Loads pre-computed stats JSONs.
"""

import os
import json
import streamlit as st


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


@st.cache_data
def load_nombres_stats():
    """Load pre-computed nombre statistics."""
    path = os.path.join(DATA_DIR, 'nombres_stats.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_apellidos_stats():
    """Load pre-computed apellido statistics."""
    path = os.path.join(DATA_DIR, 'apellidos_stats.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_training_summary():
    """Load training summary with model comparison metrics."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'training_summary.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
