'''

Esse arquivo implementa a interface de perguntas e respostas RAG usando Streamlit.

Ele segue o seguinte pipeline:
    - Carrega o índice vetorial criado pelo ingest.py (através do vectorstore_utils.py)
    - Aplica embeddings na pergunta do usuário com o setence-transformer em ingest.py
    - Recupera, com FAISS, os trechos mais relevantes para sugerir uma resposta

Os principais arquivos do projeto seguem o seguinte fluxo:
    - vectorstore_utils.py -> Criação de funções para salvar e carregar o banco vetorizado
    - ingest.py -> Realiza a leitura dos documentos, faz a chunkenização, faz o embedding e utiliza funções criadas no arquivo acima para guadar esses vetores de maneira indexada
    - app.py -> Usa esse índice para encontrar contexto e ajudar a responder perguntas do usuário
    
'''

# Biblioteca para armazenar e maniplar caminho de arquivos. Boa prática fundamental para garantir consistência. 
from pathlib import Path 

# Framework feito para criar páginas web simples de projetos com python
import streamlit as st

# LangChain é um framework orquestrados, ou seja, ele que irá conectar LLM, banco vetorizado, Embeddings, pergunta do user, etc
# HuggingFaceEmbeddings é a classe que possui funções e modelos para realizar o processo de embedding. 
from langchain_huggingface import HuggingFaceEmbeddings

# vectorstore_utils é uma biblioteca criada localmente com funções úteis para utilizar o FAISS da melhor maneira
# load_faiss_store é uma função com o objetivo de carregar os índices do banco vetorizado, ou seja, será utilizada quando for necessário 
#     consultar o banco para comparar os chunks da pergunta do usuário com os chunks dos arquivos utilizados para compôr o banco de dados
from vectorstore_utils import load_faiss_store

# Define os caminhos de onde o banco vetorizado será carregado
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# A linha 37 permite que o banco vetorizado seja carregado apenas uma vez por sessão, gerando ganho de performance 
@st.cache_resource

# Função para carregar os
def carregar_base():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2" # Utiliza os mesmo modelo do ingest.py e isso é fundamental para o projeto não apresentar erros
    )
    vectorstore = load_faiss_store(VECTORSTORE_DIR, embeddings) # Recupera o íncide FAISS que foi salvo previamente 
    
    return vectorstore # Retorna o banco vetorial pronto para buscar informações semelhantes à pergunta do usuário

# Função responsável por montar a resposta à pergunta do usuário
# Essa função receberá a pergunta do usuário e os documentos retornado pelo FAISS.
#     Contexto: junto ao conteúdo textual dos chunks recuperados como base/contexto da resposta
#     Fontes: Identifica e lista o nome dos arquivos originais das respostas encontradas
# Retorna um texto consolidado com pergunta, contexto, instrução para resumir com a base nesse contexto e as fontes
def montar_resposta(pergunta: str, docs):

    # Junta o conteúdo textual de todos os documentos recuperados em um único bloco de texto, separado por linhas em branco
    # Por exemplo, pega os três trechos mais relevantes e coloca eles juntos de maneira organizada
    # Isso forma o contexto a ser utilizado na resposta
    contexto = "\n\n".join([doc.page_content for doc in docs]) 

    # Monta uma lista única das fontes originais dos trechos recuperados
    # Usa os metadados de cada doumento, doc.metadata["source"], e extrai só o nome do arquivo ordenando em ordem alfabética.
    # Se por algum motivo a fonte não existir, aparecerá "desconhecida"
    fontes = sorted({Path(doc.metadata.get("source", "desconhecida")).name for doc in docs})

    # Da linha 66 até a 76 é utilizada uma f-string para montar a resposta final ao usuário
    # A formatação cria um texto dizendo:
    #    - Pergunta feita pelo usuário
    #    - O contexto recuperado dos documentos
    #    - Uma instrução explícita 
    #    - As fontes utilizadas para criar a resposta
    
    resposta = f"""
Pergunta: {pergunta}

Contexto recuperado:
{contexto}

Resumo da resposta:
Com base nos trechos recuperados, a resposta deve ser construída usando apenas as informações acima.

Fontes: {", ".join(fontes)}
"""
    return resposta

# Configurações estéticas da página
st.set_page_config(page_title="RAG Básico", page_icon="📚")
st.title("📚 Assistente com recuperação semântica")
st.write("Faça uma pergunta sobre os documentos carregados.")

# Local onde o usuário irá fazer a pergunta
pergunta = st.text_input(
    "Pergunta",
    placeholder="Ex.: O que é RAG e qual o papel do FAISS?",
)

'''
 Botão para iniciar a consulta
 
 Primeiro: Validação para garantir que o campo não está vázio
 
 Segundo: Carregamento do index FAISS (banco vetorizado) 
 
 Terceiro: Criação do retriver
      - Retriver é um objeto que, para cada pergunta, retorna os 3 chunks mais próximos a pergunta do usuário
 
 Quarto: Realiza a busca
     - Passa a pergunta pro retriver
     - O index FAISS encontra os textos mais próximos
     - Retorna para uso

 Quinto: Exibição dos trechos
     - Mostra para o usuário o contexto trabalhado para a resposta
     - Para cada documento encontrado:
         - Mostra o texto do chunk
         - Mostra o nome do arquivo original
         
 Sexto: Resposta montada
'''
if st.button("Buscar contexto"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta.") # Primeiro
    else:
        vectorstore = carregar_base() # Segundo
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Terceiro
        docs = retriever.invoke(pergunta) # Quarto

        st.subheader("Trechos recuperados") # Quinto
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Trecho {i}**")
            st.write(doc.page_content)
            st.caption(f"Fonte: {doc.metadata.get('source', 'desconhecida')}")

        st.subheader("Resposta montada") #Sexto
        st.code(montar_resposta(pergunta, docs))
