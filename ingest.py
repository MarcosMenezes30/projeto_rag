'''
Esse arquivo tem o objetivo de :
    - Ler os arquivos
    - Processá-los em vetores através de Embedding
    - Colocar tudo num banco vetorial indexado (FAISS), pronto para busca
'''

# Biblioteca para armazenar e manipular caminho de arquivos. Boa prática fundamental. 
from pathlib import Path


# LangChain é um framework orquestrador, ou seja, ele que irá conectar LLM, FAISS, Embeddings, etc

# Realiza o processo de chunkerização, ou seja, divide o texto em frações menores para serem buscados e processador de maneira mais eficiente
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Escaneia uma pasta inteira e carrega todos arquivos, como arquivos .txt, de uma vez só
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Motor do banco vetorial, ou seja, quem irá realizar toda a montagem e indexação do banco de textos processados (embeddings)
from langchain_community.vectorstores import FAISS

# Gera embeddings dos pecaços de texto, ou seja, realiza um provesso de vetorização eficiente que leva em consideração contexto e valor semântico das palavras
from langchain_huggingface import HuggingFaceEmbeddings


# Função criada localmente no arquivo "vectorstore_utils.py" com o objetivo de salvar a vetorização feita em disco local. 
# Isso facilita a reutilização e mantém o projeto DRY (Don't Repeat Yourself)
from vectorstore_utils import save_faiss_store

# As linhas 30 a 32 basicamente estão armazenando caminhos para determinados arquivos em variáveis. Boa prática fundamental que garante consistência.
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs" # local onde os arquivos que serão indexados estão
VECTORSTORE_DIR = BASE_DIR / "vectorstore" # local onde o banco de vetores vai ficar salvo, para ser carregado no app.py depois

# Função principal deste arquivo onde todo o processamento do pipeline de ingestão ocorrerá
def main() -> None:

    # Aqui estamos instanciando a função DirectoryLoader em loader
    # A função DirectoryLoader faz um scan recursivo em docs/, carregando todos os arquivos .txt (porque definimos isso na linha 40)
    # ⚠️ Cada arquivo lido é transformado em um objeto de documento do LangChain, padronizado para fluxo ⚠️
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load() # Por fim, a chamamos

    # Aqui estamos instanciando a função RecursiveCharacterTextSplitter em splitter
    # A função RecursiveCharacterTextSplitter tem o objetivo de chunkinizar textos para melhorar seu processamento, os dividindo em pedaços menores
    splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=400, # Definição que cada chunk terá no máximo 400 carateres 
        
        chunk_overlap=80, # Os chunks se sobrepõem 80 caracteres entre eles. Isso ocorre para garantir que não tenha nenhuma informação importante ou o contexto que se perca na divisão dos chunks
        
    )
    chunks = splitter.split_documents(documents) # Por fim a chamamos com documents em seu argumento, ou seja, todos os documentos txt presentes em docs/ serão chunkinizados

    # Aqui estamos instanciando a função HuggingFaceEmbeddings em embeddings
    # A função HuggingFaceEmbeddings tem o objetivo de realizar o processo de Embedding (vetorização otimizada) nos chunks processados na etapa anterior
    # Sentence-transformers são modelos de ML desenvolvidos especificamente para transformar sentenças em Embeddings.
    # O modelo all-MiniLM-L6-v2 é um exemplo que foi utilizado nesse projeto pois é um ótimo equilíbrio entre velocidade, qualidade e tamanho
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2" 
    )

    # Aqui estaremos construindo o banco vetorial (vectorstore FAISS) para facilitar a busca rápida de chunks semelhantes a pergunta do usuário
    # Essa função criará o banco vetorial a partir dos chunks que passarão pelo processo de vetorização. Por isso chunks e embeddings estão como parâmetros dessa função
    # Será aqui que o processo de Embedding ocorrerá e, ao chunk ser vetorizado, ele será guardado no banco vetorial com um índice e metadados do chunk
    # Tudo isso tornará possível que o embedding da pergunta do usuário seja comparada com todos os embeddings do banco vetorial de maneira rpaida, assertiva e eficiente
    vectorstore = FAISS.from_documents(chunks, embeddings) 

    # Depois que o banco vetorial foi criado, ele irá salvar o índice inteiro no disco.
    # Desta maniera, em futuras execuções o acesso será feito diretamente no disco, tornanto o processo mais veloz
    output_path = save_faiss_store(vectorstore, VECTORSTORE_DIR)

    # Prints para acompanhamento com feedback de quantos arquivos foram processador, quantos chunks foram criados e onde ficou salvo.
    print(f"Documentos carregados: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(f"Base vetorial salva em: {output_path}")


if __name__ == "__main__":
    main()
