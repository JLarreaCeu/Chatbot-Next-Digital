# Práctica final - Jorge Larrea
# Chatbot con RAG, búsqueda web y guardarraíles

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

load_dotenv()

RUTA_PDF = os.path.join(os.path.dirname(__file__), "datos", "documento.pdf")


@st.cache_resource
def cargar_agente():
    # sin cache_resource esto re-indexaría el PDF en cada mensaje del usuario
    loader = PyPDFLoader(RUTA_PDF)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="practica_final")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # el docstring de cada tool es lo que el agente lee para decidir cuándo usarla
    @tool
    def buscar_en_documento(consulta: str) -> str:
        """Busca en el PDF indexado. Usa esta herramienta primero para cualquier pregunta."""
        resultados = retriever.invoke(consulta)
        if not resultados:
            return "No encontré nada relevante en el documento."
        return "[Fuente: documento]\n\n" + "\n\n".join(d.page_content for d in resultados)

    ddg = DuckDuckGoSearchRun()

    @tool
    def buscar_en_internet(consulta: str) -> str:
        """Busca en internet. Úsala solo si el documento no tiene la respuesta."""
        try:
            return "[Fuente: internet]\n\n" + ddg.run(consulta)
        except Exception as e:
            return f"No pude buscar en internet: {e}"

    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

    agente = create_react_agent(
        llm,
        tools=[buscar_en_documento, buscar_en_internet],
        prompt=(
            "Eres un asistente especializado en el tema del documento que tienes indexado. "
            "Cuando alguien te pregunte algo, primero busca en el documento. "
            "Si ahí no está la respuesta o necesita datos actualizados, busca en internet. "
            "Deja claro de dónde viene la información."
        ),
    )
    return agente, llm


def es_relevante(pregunta, llm):
    # le pregunto al modelo si la pregunta tiene que ver con el tema antes de llamar al agente
    respuesta = llm.invoke([
        {
            "role": "system",
            "content": (
                "Clasifica la pregunta. Responde solo con 'relevante' o 'irrelevante'.\n\n"
                "Es relevante si habla de inteligencia artificial, machine learning, LLMs, "
                "redes neuronales, o si pregunta sobre este chatbot o el documento que usa. "
                "Los saludos también son relevantes.\n\n"
                "Es irrelevante si pregunta algo completamente ajeno, como recetas, deportes o política."
            ),
        },
        {"role": "user", "content": pregunta},
    ])
    return "irrelevante" not in respuesta.content.strip().lower()


def ejecutar_agente(agente, historial):
    pasos = []
    respuesta = ""

    # stream nos da los pasos intermedios para mostrarlos en el expander
    for chunk in agente.stream({"messages": historial}, stream_mode="updates"):
        for nodo, salida in chunk.items():
            if "messages" not in salida:
                continue
            for m in salida["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        args = tc.get("args", {})
                        q = list(args.values())[0] if args else ""
                        pasos.append(f"**Herramienta:** `{tc.get('name')}` — *\"{q}\"*")
                elif nodo == "tools" and hasattr(m, "content") and m.content:
                    texto = str(m.content)
                    # recortamos para no llenar el expander
                    pasos.append(f"**Resultado:** {texto[:400]}{'...' if len(texto) > 400 else ''}")
                elif nodo == "agent" and hasattr(m, "content") and m.content:
                    respuesta = m.content

    return respuesta, pasos


def main():
    st.set_page_config(page_title="Chatbot JL", page_icon="C", layout="centered")
    st.title("Chatbot sobre IA")
    st.caption("Busca en el documento y en internet si hace falta")

    with st.sidebar:
        st.header("Opciones")
        ver_pasos = st.toggle("Ver qué hace el agente por dentro", value=False)
        st.divider()
        st.info("Pregúntame cualquier cosa sobre inteligencia artificial. Si no está en el documento lo busco en internet.")
        if st.button("Borrar conversación", use_container_width=True):
            st.session_state.mensajes = []
            st.rerun()

    # session_state guarda el historial entre recargas de streamlit
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    agente, llm = cargar_agente()

    if pregunta := st.chat_input("¿Qué quieres saber?"):
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            if not es_relevante(pregunta, llm):
                # el guardarraíl bloqueó la pregunta, no llamo al agente
                respuesta = "Eso está fuera de mi tema. Solo puedo ayudarte con preguntas sobre inteligencia artificial y cosas relacionadas."
                st.markdown(respuesta)
            else:
                # convierto el historial al formato que usa langchain
                historial = []
                for msg in st.session_state.mensajes:
                    if msg["role"] == "user":
                        historial.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        historial.append(AIMessage(content=msg["content"]))

                with st.spinner("Buscando..."):
                    respuesta, pasos = ejecutar_agente(agente, historial)

                if ver_pasos and pasos:
                    with st.expander("Ver pasos del agente"):
                        for paso in pasos:
                            st.markdown(paso)

                st.markdown(respuesta)

        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})


if __name__ == "__main__":
    main()
