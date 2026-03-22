"""
MakeMore Argentina — Generador de Nombres y Apellidos Argentinos con IA
Main Streamlit app entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="MakeMore Argentina",
    page_icon="🇦🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("MakeMore Argentina")
st.sidebar.markdown("Generador de nombres y apellidos argentinos con inteligencia artificial")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Navegación:** usá el menú de páginas de arriba para acceder a cada sección."
)

st.title("🇦🇷 MakeMore Argentina")
st.subheader("Generador de Nombres y Apellidos Argentinos con IA")

st.markdown("""
### ¿Qué es este proyecto?

Este proyecto adapta [makemore](https://github.com/karpathy/makemore) de **Andrej Karpathy** 
para generar nombres y apellidos que *suenan* argentinos, usando datos reales del 
**Registro Nacional de las Personas (RENAPER)**.

### Datos utilizados

- **Nombres históricos:** más de 9.7 millones de registros de personas nacidas en Argentina 
  entre 1922 y 2015, con la cantidad de personas que llevaron cada nombre por año.
- **Apellidos por provincia:** más de 320.000 registros de los apellidos más frecuentes 
  en las 24 jurisdicciones argentinas (datos de diciembre 2021).

### ¿Qué podés hacer?

| Página | Descripción |
|--------|-------------|
| **Generar** | Generá nombres y/o apellidos nuevos usando diferentes modelos de IA |
| **Autocompletar** | Escribí las primeras letras y el modelo te sugiere completaciones |
| **Acerca del proyecto** | Métricas de los modelos, recursos oficiales y créditos |

### Modelos implementados

Se entrenan **6 arquitecturas** de modelos de lenguaje a nivel de carácter, 
desde la más simple hasta la más avanzada:

1. **Bigram** — tabla de probabilidades de pares de caracteres
2. **MLP** — red neuronal feedforward (Bengio et al. 2003)
3. **BoW** — Bag of Words con atención causal
4. **RNN** — red neuronal recurrente
5. **GRU** — Gated Recurrent Unit
6. **Transformer** — arquitectura GPT-2

Cada modelo aprende los patrones de caracteres de los nombres y apellidos argentinos 
y puede generar nuevos que respetan esos patrones.
""")

st.markdown("---")
st.caption("Proyecto educativo basado en makemore de Andrej Karpathy. Datos: RENAPER Argentina.")
