"""
Módulo de Ingesta (Setup)

Lee archivos .txt/.md de la carpeta /data, los fragmenta con
RecursiveCharacterTextSplitter y persiste los chunks en ChromaDB.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

VECTORSTORE_PATH = "./vectorstore"
COLLECTION_NAME = "rag_documents"
DATA_PATH = "./data"

# Chunking: ~500 tokens con 50 de overlap usando cl100k_base
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings():
    """Retorna el modelo de embeddings según LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    # Default: Ollama con nomic-embed-text
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def collection_exists() -> bool:
    """Verifica si ya existe una colección poblada en ChromaDB."""
    if not Path(VECTORSTORE_PATH).exists():
        return False
    try:
        client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def load_documents(data_path: str = DATA_PATH):
    """Carga todos los archivos .txt y .md del directorio de datos."""
    docs = []
    for glob_pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            data_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs.extend(loader.load())
    print(f"  Documentos cargados: {len(docs)}")
    return docs


def chunk_documents(documents):
    """Fragmenta documentos con RecursiveCharacterTextSplitter basado en tokens."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Chunks generados: {len(chunks)} (chunk_size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP})")
    return chunks


def ingest_documents(data_path: str = DATA_PATH, force: bool = False):
    """
    Pipeline completo de ingesta.

    Si la colección ya existe y force=False, retorna el vectorstore existente
    sin volver a indexar (evita costos y tiempo innecesarios).
    """
    from langchain_chroma import Chroma

    if collection_exists() and not force:
        print("Vectorstore existente detectado. Cargando sin re-indexar.")
        return Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )

    print("=== Iniciando ingesta de documentos ===")
    documents = load_documents(data_path)

    if not documents:
        raise ValueError(f"No se encontraron documentos en '{data_path}'")

    chunks = chunk_documents(documents)
    embeddings = get_embeddings()

    print("  Generando embeddings y persistiendo en ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
        collection_name=COLLECTION_NAME,
    )
    print(f"  Vectorstore creado en '{VECTORSTORE_PATH}'")
    print("=== Ingesta completada ===\n")
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta de documentos en ChromaDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la re-ingesta aunque el vectorstore ya exista",
    )
    args = parser.parse_args()

    vectorstore = ingest_documents(force=args.force)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("Retriever listo. Ejemplo de búsqueda:")
    results = retriever.invoke("¿Qué es el aprendizaje supervisado?")
    for i, doc in enumerate(results, 1):
        src = Path(doc.metadata.get("source", "")).name
        print(f"  [{i}] {src}: {doc.page_content[:120]}...")
