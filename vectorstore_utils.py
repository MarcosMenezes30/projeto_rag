from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


VECTORSTORE_FILENAME = "index.pkl"


def save_faiss_store(vectorstore: FAISS, folder_path: Path) -> Path:
    """Persist the vector store without relying on FAISS file-path handling."""
    folder_path.mkdir(parents=True, exist_ok=True)
    output_path = folder_path / VECTORSTORE_FILENAME
    output_path.write_bytes(vectorstore.serialize_to_bytes())
    return output_path


def load_faiss_store(folder_path: Path, embeddings: Embeddings) -> FAISS:
    """Load a vector store previously saved by save_faiss_store."""
    input_path = folder_path / VECTORSTORE_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(
            f"Base vetorial nao encontrada em: {input_path}. Rode `python ingest.py` antes."
        )

    return FAISS.deserialize_from_bytes(
        input_path.read_bytes(),
        embeddings,
        allow_dangerous_deserialization=True,
    )
