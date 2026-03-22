"""
Página: Generar Nombres y Apellidos
Genera nombres y/o apellidos argentinos usando modelos entrenados.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import get_available_models, load_model_cached
from makemore_ar import generate_names, MODEL_DESCRIPTIONS, MODEL_FRIENDLY_INFO

st.set_page_config(page_title="Generar — MakeMore Argentina", page_icon="🇦🇷", layout="wide")

st.title("✨ Generar Nombres y Apellidos")
st.markdown("Generá nombres o apellidos argentinos nuevos usando modelos de IA entrenados con datos del RENAPER.")

# --- Dataset selection ---
col1, col2 = st.columns(2)
with col1:
    dataset_type = st.radio(
        "¿Qué querés generar?",
        ["Nombre", "Apellido", "Nombre completo (nombre + apellido)"],
        index=0,
    )

# Determine which models to load
datasets_needed = []
if dataset_type == "Nombre":
    datasets_needed = ["nombres"]
elif dataset_type == "Apellido":
    datasets_needed = ["apellidos"]
else:
    datasets_needed = ["nombres", "apellidos"]

# Check available models
all_available = {}
for ds in datasets_needed:
    available = get_available_models(ds)
    if available:
        all_available[ds] = available

if not all_available:
    st.warning(
        "⚠️ No se encontraron modelos entrenados. "
        "Ejecutá `python train.py` primero para entrenar los modelos."
    )
    st.code("python train.py --dataset nombres --model-type bigram --max-steps 5000", language="bash")
    st.stop()

# --- Model selection ---
# Use the first available dataset to determine model types
first_ds = list(all_available.keys())[0]
available_types = list(all_available[first_ds].keys())

with col2:
    model_type = st.selectbox(
        "Modelo",
        available_types,
        format_func=lambda x: MODEL_DESCRIPTIONS.get(x, x),
    )

# --- Generation parameters ---
st.markdown("---")
col_a, col_b, col_c = st.columns(3)
with col_a:
    num_samples = st.slider("Cantidad a generar", min_value=1, max_value=50, value=10)
with col_b:
    temperature = st.slider("Temperatura", min_value=0.1, max_value=2.0, value=0.8, step=0.1,
                            help="Más bajo = más conservador, más alto = más creativo")
with col_c:
    top_k_val = st.slider("Top-K", min_value=0, max_value=50, value=0,
                          help="0 = sin filtro. Limita a las K opciones más probables")

top_k = top_k_val if top_k_val > 0 else None

# --- Generate ---
if st.button("🎲 Generar", type="primary", use_container_width=True):
    results_by_ds = {}

    for ds in datasets_needed:
        if ds not in all_available or model_type not in all_available[ds]:
            st.warning(f"No hay modelo {model_type} entrenado para {ds}.")
            continue

        checkpoint_path = all_available[ds][model_type]
        with st.spinner(f"Cargando modelo {model_type} para {ds}..."):
            model, dataset = load_model_cached(checkpoint_path)

        with st.spinner(f"Generando {ds}..."):
            generated = generate_names(model, dataset, num_samples=num_samples,
                                       temperature=temperature, top_k=top_k)
        results_by_ds[ds] = generated

    # Display results
    st.markdown("---")
    st.subheader("Resultados")

    if dataset_type == "Nombre completo (nombre + apellido)" and len(results_by_ds) == 2:
        nombres = results_by_ds.get("nombres", [])
        apellidos = results_by_ds.get("apellidos", [])

        # Pair them up
        pairs = list(zip(nombres, apellidos))
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("**Nombres completos generados:**")
            for (nombre, n_cat), (apellido, a_cat) in pairs:
                full_name = f"{nombre.title()} {apellido.title()}"
                badges = []
                if n_cat == 'novel':
                    badges.append("nombre novel")
                if a_cat == 'novel':
                    badges.append("apellido novel")
                badge_str = f" _({', '.join(badges)})_" if badges else ""
                st.markdown(f"- **{full_name}**{badge_str}")

        with col_res2:
            st.markdown("**Detalle:**")
            novel_count = sum(1 for (_, nc), (_, ac) in pairs if nc == 'novel' or ac == 'novel')
            existing_count = len(pairs) - novel_count
            st.metric("Novel (al menos 1 parte nueva)", novel_count)
            st.metric("Ambas partes existentes", existing_count)
    else:
        for ds, generated in results_by_ds.items():
            ds_label = "Nombres" if ds == "nombres" else "Apellidos"
            st.markdown(f"**{ds_label} generados:**")

            novel = [(w, c) for w, c in generated if c == 'novel']
            existing = [(w, c) for w, c in generated if c == 'existente']

            col_n, col_e = st.columns(2)
            with col_n:
                st.markdown(f"🆕 **Nuevos** ({len(novel)})")
                for word, _ in novel:
                    st.markdown(f"- {word.title()}")
            with col_e:
                st.markdown(f"📋 **Existentes en dataset** ({len(existing)})")
                for word, _ in existing:
                    st.markdown(f"- {word.title()}")
