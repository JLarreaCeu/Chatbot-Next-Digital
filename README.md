<div align="center">
  <h1>Chatbot sobre IA</h1>
  <p>Práctica final del taller de agentes - Next Digital</p>
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="python">
  <img src="https://img.shields.io/badge/LangChain-0.3+-green" alt="langchain">
  <img src="https://img.shields.io/badge/Streamlit-1.57+-red" alt="streamlit">
</div>

---

Un chatbot que responde preguntas sobre un documento PDF. Primero busca en el documento, y si no encuentra la respuesta, busca en internet. Si la pregunta no tiene nada que ver con el tema del documento, el guardarraíl la bloquea antes de llegar al agente.

## Cómo funciona

El flujo es bastante directo:

1. El usuario escribe una pregunta
2. Un clasificador (el guardarraíl) decide si la pregunta es relevante para el tema del documento
3. Si lo es, el agente ReAct busca primero en el PDF indexado en Chroma
4. Si el documento no tiene la respuesta, el agente busca en internet con DuckDuckGo
5. La respuesta siempre indica de dónde viene la información

Hay una opción en el sidebar para ver los pasos que va dando el agente por dentro, lo que ayuda bastante a entender cómo razona.

## Stack

- Python 3.11
- LangChain + LangGraph (`create_react_agent`)
- OpenAI `gpt-5.4-nano` para el agente y el clasificador
- OpenAI `text-embedding-3-small` para los embeddings
- Chroma como base de datos vectorial
- DuckDuckGo para búsqueda web
- Streamlit para la interfaz

## Instalación

```bash
git clone https://github.com/KrembolCeu/Proyecto-Chatbot-Next-Digital.git
cd Proyecto-Chatbot-Next-Digital

# instalar dependencias
uv sync

# copiar el archivo de entorno y añadir la API key
cp .env.example .env
```

Edita `.env` y añade tu `OPENAI_API_KEY`.

También necesitas tener el PDF que quieras indexar en `../taller-agentes-universidades/datos/documento.pdf` (o cambiar la ruta `RUTA_PDF` en `app.py`).

## Ejecución

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run streamlit run app.py
```

La variable de entorno `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` es necesaria por un conflicto entre la versión de protobuf y chromadb en Python 3.14.

## Estructura

```
app.py          # toda la lógica: agente, guardarraíl e interfaz
pyproject.toml  # dependencias
.env.example    # variables de entorno necesarias (sin valores reales)
```

## Variables de entorno

Copia `.env.example` a `.env` y rellena los valores:

```
OPENAI_API_KEY=tu-api-key-aqui
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```
