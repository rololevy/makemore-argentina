"""
Página: Autocompletar
Dado un prefijo (letras o sílaba), el modelo genera completaciones.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import get_available_models, load_model_cached
from makemore_ar import generate_completions, MODEL_DESCRIPTIONS, MODEL_FRIENDLY_INFO

st.set_page_config(page_title="Autocompletar — MakeMore Argentina", page_icon="🇦🇷", layout="wide")

st.title("🔤 Autocompletar")
st.markdown(
    "Escribí las primeras letras de un nombre o apellido y el modelo te sugiere completaciones."
)

# --- Controls ---
col1, col2 = st.columns(2)
with col1:
    dataset_type = st.radio("Tipo", ["Nombre", "Apellido"], index=0, horizontal=True)
with col2:
    ds_name = "nombres" if dataset_type == "Nombre" else "apellidos"
    available = get_available_models(ds_name)
    if not available:
        st.warning(f"No hay modelos entrenados para {ds_name}. Ejecutá `python train.py` primero.")
        st.stop()
    model_type = st.selectbox(
        "Modelo",
        list(available.keys()),
        format_func=lambda x: MODEL_DESCRIPTIONS.get(x, x),
    )
    st.caption(MODEL_FRIENDLY_INFO.get(model_type, ''))

prefix = st.text_input(
    "Escribí las primeras letras:",
    value="mar",
    max_chars=30,
    help="Ingresá una letra, sílaba, o el comienzo de un nombre/apellido",
).strip().lower()

with st.expander("ℹ️ ¿Qué significan estos parámetros?"):
    st.markdown(
        "La **temperatura** y el **Top-K** controlan cómo genera el modelo. "
        "Con los valores por defecto ya funciona bien — solo modificalos si querés experimentar."
    )

col_a, col_b, col_c = st.columns(3)
with col_a:
    num_completions = st.slider("Cantidad de sugerencias", 5, 50, 20)
with col_b:
    temperature = st.slider("Temperatura", 0.1, 2.0, 0.8, 0.1,
                            help="Controla qué tan 'arriesgado' es el modelo. "
                                 "Bajo (0.1-0.5): nombres más comunes y predecibles. "
                                 "Alto (1.0-2.0): nombres más originales pero pueden sonar raros.")
with col_c:
    top_k_val = st.slider("Top-K", 0, 50, 0,
                          help="Limita cuántas opciones considera el modelo para la próxima letra. "
                               "Bajo (3-10): resultados más seguros. "
                               "0: sin límite, considera todas las opciones.")

top_k = top_k_val if top_k_val > 0 else None

# --- Generate completions ---
if prefix and st.button("🔍 Autocompletar", type="primary", use_container_width=True):
    checkpoint_path = available[model_type]

    with st.spinner("Cargando modelo..."):
        model, dataset = load_model_cached(checkpoint_path)

    # Check if prefix chars are in vocabulary
    unknown_chars = [ch for ch in prefix if ch not in dataset.stoi]
    if unknown_chars:
        st.error(f"Caracteres no reconocidos por el modelo: {unknown_chars}")
        st.stop()

    with st.spinner("Generando completaciones..."):
        completions = generate_completions(
            model, dataset, prefix,
            num_samples=num_completions * 3,  # generate extra, then deduplicate
            temperature=temperature,
            top_k=top_k,
        )

    # Deduplicate and sort
    unique_completions = list(dict.fromkeys(completions))[:num_completions]

    st.markdown("---")
    st.subheader(f"Completaciones para \"{prefix}\"")

    # Separate into existing and novel
    existing = [w for w in unique_completions if dataset.contains(w)]
    novel = [w for w in unique_completions if not dataset.contains(w)]

    col_n, col_e = st.columns(2)

    with col_n:
        st.markdown(f"🆕 **Nuevos** ({len(novel)})")
        for word in novel:
            st.markdown(f"- {word.title()}")

    with col_e:
        st.markdown(f"📋 **Existentes en dataset** ({len(existing)})")
        for word in existing:
            st.markdown(f"- {word.title()}")

    if not unique_completions:
        st.info("No se generaron completaciones válidas. Probá con otro prefijo o aumentá la temperatura.")

elif not prefix:
    st.info("Ingresá al menos una letra para autocompletar.")
