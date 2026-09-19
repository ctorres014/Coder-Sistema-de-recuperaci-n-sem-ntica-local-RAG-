"""
Script principal de demostración del sistema RAG.

Ejecuta dos pruebas:
  1. Pregunta cuya respuesta SÍ está en los documentos
  2. Pregunta trampa cuya respuesta NO está en los documentos
"""

import asyncio
import textwrap
from pathlib import Path

from ingest import ingest_documents
from rag_chain import TOP_K, build_rag_chain, get_rag_response


SEPARATOR = "=" * 65


def print_response(label: str, query: str, response):
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    print(f"Pregunta : {query}")
    print(f"Confianza: {response.confidence}")
    print(f"\nRespuesta:\n{textwrap.fill(response.answer, width=65)}")
    if response.sources:
        print(f"\nFragmentos de contexto usados ({len(response.sources)}):")
        for i, src in enumerate(response.sources, 1):
            snippet = src[:120].replace("\n", " ")
            print(f"  [{i}] {snippet}...")
    print(SEPARATOR)


async def run_demo():
    # ---- Paso 1: Ingesta ------------------------------------------------
    print("\n[1/3] Verificando / poblando vectorstore...")
    vectorstore = ingest_documents()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # ---- Paso 2: Construir cadena RAG -----------------------------------
    print("[2/3] Construyendo cadena RAG (LCEL)...")
    chain = build_rag_chain(retriever)

    # ---- Paso 3: Pruebas ------------------------------------------------
    print("[3/3] Ejecutando pruebas...\n")

    # Pregunta 1: respuesta DENTRO del contexto
    q1 = "¿Qué es el mecanismo de atención (attention) en los Transformers y cómo se calcula?"
    r1 = await get_rag_response(q1, retriever, chain)
    print_response("PRUEBA 1 — Respuesta en los documentos", q1, r1)

    # Pregunta 2: pregunta trampa (respuesta NO está en los documentos)
    q2 = "¿Cuál es la receta tradicional de la paella valenciana y cuántos ingredientes lleva?"
    r2 = await get_rag_response(q2, retriever, chain)
    print_response("PRUEBA 2 — Pregunta trampa (fuera del contexto)", q2, r2)

    # Verificación automática
    print("\n--- Verificación ---")
    ok1 = r2.confidence == "sin_informacion" or "no tengo" in r2.answer.lower() or "no" in r2.answer.lower()
    print(f"¿El modelo reconoció que no tiene info para P2? {'✓ Sí' if ok1 else '✗ No — revisar el prompt'}")


if __name__ == "__main__":
    asyncio.run(run_demo())
