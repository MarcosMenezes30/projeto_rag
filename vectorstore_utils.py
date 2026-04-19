'''
No RAG, o "vectorstore" é uma base de dados de vetores que representam textos transformados (embeddings).
Serve para achar rapidamente informações relevantes via busca semântica.

O FAISS, usado aqui, é o motor que faz essas buscas de maneira eficiente, mesmo se forem centenas de milhares de documentos

Aqui, este arquivo salva um index FAISS em disco para reaproveitar depois (evitando reprocessamento) e carregar usando os mesmos embeddings quando queremos consultar via IA.
'''

# Biblioteca que permite armazenar e manipular caminho para arquivos. Boa prática importante para garantir consistência.
from pathlib import Path

# LangChain é um framework orquestrador, ou seja, ele irá conectar LLM, contexto, chunks, etc
# FAISS é uma clase que representa o index (Baanco de vetores), resposável por gravar, ler e buscar os embeddings
from langchain_community.vectorstores import FAISS

# Embedding é a classe que torna possível um processo de vetorização mais eficaz, levando em consideração o contexto e valor semântico das palavras.
from langchain_core.embeddings import Embeddings

# Essas bibliotecas serão utilizadas serão utilizadas tanto nesse arquivo quanto no ingest.py e app.py

# Essa constante com o nome do arquivo define o nome padrão de onde a base de vetores será salva/carregada
VECTORSTORE_FILENAME = "index.pkl"

# Essa função serve para, depois de criar/preencher o vectorstore, essa função seja utilizada salvar o banco de vetores, assim não é necessário carregar tudo de novo depois
def save_faiss_store(vectorstore: FAISS, folder_path: Path) -> Path:
    """Persist the vector store without relying on FAISS file-path handling."""
    folder_path.mkdir(parents=True, exist_ok=True) # Cria uma pasta para salvar o banco caso essa pasta ainda NÃO exista
    output_path = folder_path / VECTORSTORE_FILENAME # Define o caminho completo do arquivo
    output_path.write_bytes(vectorstore.serialize_to_bytes()) # Salva o estado inteiro do banco em binário
    return output_path # Retorna o caminho do arquivo gerado para outros cripts poderem usar 

# Essa função será utilizada no ingest.py no final do arquivo, depois de montar o vectorstore pegando todos os textos e fazendo o embedding neles

# Essa função serve para, antes de responder perguntas com IA usando RAG, e necessário carregar o banco de vetores que estava salvo no passo anterior
def load_faiss_store(folder_path: Path, embeddings: Embeddings) -> FAISS:
    """Load a vector store previously saved by save_faiss_store."""
    input_path = folder_path / VECTORSTORE_FILENAME # Monta o caminho do arquivo
    if not input_path.exists(): # Verifica se o caminho existe. Importante pois, caso não exista, dá um erro guiado para o usuário saber que precisa gerar o banco primeiro
        raise FileNotFoundError(
            f"Base vetorial nao encontrada em: {input_path}. Rode `python ingest.py` antes."
        )
    # Caso exista, ele vem para essa parte da função. Aqui o arquivo é aberto, os bytes são lidos e o objeto FAISS é reconstruído
    return FAISS.deserialize_from_bytes(
        input_path.read_bytes(),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    '''
    Essa função costuma ser chamada no app.py, pois carregando o vectorstore é possível que para cada pergunta do usuário, 
    o sistema busque os textos mais parecidos/relevantes nos dados originais e, a partir isso, o LLM monte a resposta.
    
    Fluxo prático RAG nesse projeto:

    1. Ingestão (ingest.py)
        Nele os textos são lidos, transformados em embeddings, injetados no FAISS e salvos usando load_faiss_store

    2. Serviço (app.py)
        Nele os vetores são chamados usando load_faiss_store e, para cada pergunta, o modelo:
        
        busca similaridade -> monta contexto -> gera resposta
    
    '''
