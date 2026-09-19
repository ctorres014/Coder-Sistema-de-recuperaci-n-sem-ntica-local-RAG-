"""
Lanza una pregunta individual al sistema RAG desde la terminal.

Uso:
    python ask.py "¿Qué es el aprendizaje supervisado?"
    python ask.py          # modo interactivo
"""

import asyncio
import sys
import textwrap

from ingest import ingest_documents
from rag_chain import TOP_K, build_rag_chain, get_rag_response


async def ask(question: str):
    vectorstore = ingest_documents()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    chain = build_rag_chain(retriever)

    print(f"\nPregunta : {question}")
    print("Buscando en documentos...\n")

    response = await get_rag_response(question, retriever, chain)

    print("=" * 60)
    print(f"Confianza : {response.confidence}")
    print(f"\nRespuesta :\n{textwrap.fill(response.answer, width=70)}")
    if response.sources:
        print(f"\nFragmentos usados ({len(response.sources)}):")
        for i, src in enumerate(response.sources, 1):
            print(f"  [{i}] {src[:120].replace(chr(10), ' ')}...")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ingresá tu pregunta: ").strip()
        if not question:
            print("No se ingresó ninguna pregunta.")
            sys.exit(1)

    asyncio.run(ask(question))
