# 🇦🇷 MakeMore Argentina

Generador de nombres y apellidos argentinos con inteligencia artificial, basado en [makemore](https://github.com/karpathy/makemore) de Andrej Karpathy.

Entrenado con datos reales del **Registro Nacional de las Personas (RENAPER)** de Argentina.

> **[Probalo en vivo →](https://makemore-argentina.streamlit.app)** _(próximamente)_

---

## ¿Qué hace?

Genera nombres y apellidos que *suenan* argentinos pero no necesariamente existen. Usa modelos de lenguaje a nivel de carácter que aprenden los patrones de los nombres reales.

**Funcionalidades:**
- **Generar** — Creá nombres, apellidos, o nombres completos nuevos
- **Autocompletar** — Escribí las primeras letras y el modelo sugiere completaciones
- **Comparar modelos** — 6 arquitecturas desde la más simple hasta un Transformer

## Datos

| Dataset | Registros | Período | Fuente |
|---------|-----------|---------|--------|
| Nombres | 206.597 únicos (9.7M registros) | 1922–2015 | [RENAPER — Nombres](https://datos.gob.ar/dataset/otros-nombres-personas-fisicas) |
| Apellidos | 136.544 únicos (321K registros) | Dic. 2021 | [RENAPER — Apellidos](https://datos.gob.ar/dataset/renaper-registro-nacional-personas) |

Los datos incluyen caracteres hispánicos: á, é, í, ó, ú, ñ, ü, espacios (nombres compuestos) y guiones.

## Modelos

Se entrenan 6 arquitecturas de complejidad creciente sobre ambos datasets (12 modelos en total):

| Modelo | Descripción | Test Loss (nombres) | Test Loss (apellidos) |
|--------|-------------|:-------------------:|:---------------------:|
| Bigram | Tabla de probabilidades de pares de caracteres | 2.312 | 2.447 |
| BoW | Bag of Words con atención causal | 1.831 | 2.214 |
| MLP | Red neuronal feedforward ([Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)) | 1.030 | 2.072 |
| RNN | Red neuronal recurrente ([Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)) | 0.992 | 2.072 |
| GRU | Gated Recurrent Unit ([Cho et al. 2014](https://arxiv.org/abs/1409.1259)) | 0.949 | 2.013 |
| **Transformer** | Arquitectura GPT-2 ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | **0.939** | **1.970** |

> Menor loss = mejor. El Transformer logra los mejores resultados en ambos datasets.

## Instalación local

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/makemore-argentina.git
cd makemore-argentina

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias (PyTorch CPU-only)
pip install -r requirements.txt

# Ejecutar la app
streamlit run streamlit_app.py