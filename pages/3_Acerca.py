"""
Página: Acerca del Proyecto y Recursos
Métricas de modelos, links a dashboards oficiales, créditos.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_nombres_stats, load_apellidos_stats, load_training_summary
from makemore_ar import MODEL_DESCRIPTIONS

st.set_page_config(page_title="Acerca — MakeMore Argentina", page_icon="🇦🇷", layout="wide")

st.title("ℹ️ Acerca del Proyecto")

# --- Official resources ---
st.subheader("📊 Dashboards oficiales del RENAPER")
st.markdown("""
Si querés consultar estadísticas de nombres y apellidos reales de Argentina, 
el gobierno ya tiene dashboards interactivos excelentes:

- 🔗 **[Buscador de Nombres](https://nombres.datos.gob.ar)** — Consultá la evolución 
  de cualquier nombre por año, ranking y frecuencia.
- 🔗 **[Estadísticas de Nombres y Apellidos — RENAPER](https://www.argentina.gob.ar/interior/renaper/estadistica-de-poblacion/nombres-y-apellidos)** — 
  Tableros oficiales con datos actualizados.

> **Nuestro proyecto no replica esos dashboards.** El valor diferencial de MakeMore Argentina 
> es la **generación de nombres nuevos con IA** — algo que los dashboards oficiales no hacen.
""")

st.markdown("---")

# --- Model comparison ---
st.subheader("🤖 Comparación de Modelos")

summary = load_training_summary()
if summary:
    # Build comparison table
    rows = []
    for entry in summary:
        rows.append({
            "Modelo": MODEL_DESCRIPTIONS.get(entry['model_type'], entry['model_type']),
            "Dataset": entry['dataset'].title(),
            "Parámetros": f"{entry['n_params']:,}",
            "Mejor Test Loss": entry['best_test_loss'],
            "Tiempo (seg)": entry['training_time_sec'],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Show samples from best model
    st.markdown("#### Muestras generadas por cada modelo")
    for entry in summary:
        model_name = MODEL_DESCRIPTIONS.get(entry['model_type'], entry['model_type'])
        ds = entry['dataset'].title()
        with st.expander(f"{model_name} — {ds}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Existentes en dataset:**")
                for s in entry.get('sample_existing', []):
                    st.markdown(f"- {s.title()}")
            with col2:
                st.markdown("**Nuevos (generados):**")
                for s in entry.get('sample_novel', []):
                    st.markdown(f"- {s.title()}")
else:
    st.info("No hay datos de entrenamiento disponibles todavía. Ejecutá `python train.py` para generar métricas.")

st.markdown("---")

# --- Dataset stats ---
st.subheader("📈 Datos de Entrenamiento")

col1, col2 = st.columns(2)

nombres_stats = load_nombres_stats()
apellidos_stats = load_apellidos_stats()

with col1:
    st.markdown("#### Nombres")
    if nombres_stats:
        st.metric("Nombres únicos", f"{nombres_stats['total_unique_names']:,}")
        st.metric("Registros procesados", f"{nombres_stats['total_rows_processed']:,}")
        st.metric("Rango temporal", f"{nombres_stats['min_year']} — {nombres_stats['max_year']}")

        with st.expander("Top 20 nombres (todos los tiempos)"):
            for entry in nombres_stats['top_100_all_time'][:20]:
                st.markdown(f"- **{entry['nombre'].title()}**: {entry['cantidad']:,}")
    else:
        st.info("Ejecutá `python data/preprocessing.py` para generar estadísticas.")

with col2:
    st.markdown("#### Apellidos")
    if apellidos_stats:
        st.metric("Apellidos únicos", f"{apellidos_stats['total_unique_surnames']:,}")
        st.metric("Registros procesados", f"{apellidos_stats['total_rows_processed']:,}")
        st.metric("Provincias", f"{len(apellidos_stats.get('provinces', {}))}")

        with st.expander("Top 20 apellidos (nacional)"):
            for entry in apellidos_stats['top_100_national'][:20]:
                st.markdown(f"- **{entry['apellido'].title()}**: {entry['cantidad']:,}")

        with st.expander("Top 5 apellidos por provincia"):
            provinces = apellidos_stats.get('top_by_province', {})
            for prov in sorted(provinces.keys()):
                st.markdown(f"**{prov}:**")
                for entry in provinces[prov][:5]:
                    st.markdown(f"- {entry['apellido'].title()}: {entry['cantidad']:,}")
    else:
        st.info("Ejecutá `python data/preprocessing.py` para generar estadísticas.")

st.markdown("---")

# --- Technical details ---
st.subheader("🔧 Detalles Técnicos")
st.markdown("""
#### Arquitectura

Los modelos son **modelos de lenguaje a nivel de carácter** (character-level language models). 
Aprenden la distribución de probabilidad del siguiente carácter dado los anteriores. 
Para generar un nombre nuevo: empiezan con un token `<START>`, predicen carácter por carácter, 
y paran cuando predicen `<STOP>`.

#### Modelos implementados

| Modelo | Descripción | Referencia |
|--------|------------|------------|
| Bigram | Tabla de probabilidades de pares de caracteres | — |
| MLP | Red feedforward con embeddings | Bengio et al. 2003 |
| BoW | Bag of Words con atención causal | — |
| RNN | Red neuronal recurrente | Mikolov et al. 2010 |
| GRU | Gated Recurrent Unit | Cho et al. 2014 |
| Transformer | Arquitectura GPT-2 completa | Vaswani et al. 2017 |


#### Stack tecnológico

- **PyTorch** — modelos de deep learning
- **Streamlit** — interfaz web
- **Datos** — RENAPER (Registro Nacional de las Personas, Argentina)
""")

st.markdown("---")

# --- Credits ---
st.subheader("🙏 Créditos")
st.markdown("""
- **Código base:** [makemore](https://github.com/karpathy/makemore) por 
  [Andrej Karpathy](https://karpathy.ai/)
- **Adaptación y desarrollo:** [Orlando](https://github.com/rololevy/makemore)
- **Datos de nombres:** Registro histórico de nombres, datos abiertos de Argentina 
  (personas nacidas entre 1922 y 2015)
- **Datos de apellidos:** RENAPER — Apellidos más frecuentes por provincia (diciembre 2021)
- **Dashboards oficiales:** [nombres.datos.gob.ar](https://nombres.datos.gob.ar) y 
  [RENAPER Estadísticas](https://www.argentina.gob.ar/interior/renaper/estadistica-de-poblacion/nombres-y-apellidos)
""")
