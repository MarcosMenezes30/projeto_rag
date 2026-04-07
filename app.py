from pathlib import Path

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore_utils import load_faiss_store

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


@st.cache_resource

def carregar_base():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = load_faiss_store(VECTORSTORE_DIR, embeddings)
    return vectorstore


def montar_resposta(pergunta: str, docs):
    contexto = "\n\n".join([doc.page_content for doc in docs])
    fontes = sorted({Path(doc.metadata.get("source", "desconhecida")).name for doc in docs})

    resposta = f"""
Pergunta: {pergunta}

Contexto recuperado:
{contexto}

Resumo da resposta:
Com base nos trechos recuperados, a resposta deve ser construída usando apenas as informações acima.

Fontes: {", ".join(fontes)}
"""
    return resposta


st.set_page_config(page_title="RAG Básico", page_icon="📚")
st.title("📚 Assistente com recuperação semântica")
st.write("Faça uma pergunta sobre os documentos carregados.")

pergunta = st.text_input(
    "Pergunta",
    placeholder="Ex.: O que é RAG e qual o papel do FAISS?",
)

if st.button("Buscar contexto"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta.")
    else:
        vectorstore = carregar_base()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(pergunta)

        st.subheader("Trechos recuperados")
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Trecho {i}**")
            st.write(doc.page_content)
            st.caption(f"Fonte: {doc.metadata.get('source', 'desconhecida')}")

        st.subheader("Resposta montada")
        st.code(montar_resposta(pergunta, docs))
