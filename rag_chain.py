"""
Capa de Recuperación y Generación (RAG Chain)

Implementa get_rag_response(query) como función asíncrona que:
  1. Busca fragmentos relevantes en ChromaDB
  2. Construye el prompt con el contexto recuperado
  3. Llama al LLM de forma asíncrona
  4. Parsea la respuesta a RAGResponse (Pydantic)
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from models import RAGResponse

load_dotenv()

# Máximo de fragmentos recuperados (evita "Lost in the Middle" con top-k > 5)
TOP_K = 4

SYSTEM_PROMPT = """\
Eres un asistente técnico especializado. Tu única fuente de información es el CONTEXTO proporcionado a continuación.

Reglas estrictas:
1. Responde ÚNICAMENTE basándote en el CONTEXTO. No uses conocimiento externo.
2. Si la respuesta no está en el CONTEXTO, responde indicando que no tienes acceso a esa información.
3. Incluye en "sources" las citas textuales más relevantes del contexto que sustenten tu respuesta.
4. El campo "confidence" debe ser "sin_informacion" cuando la respuesta no esté en el contexto.

{format_instructions}

CONTEXTO:
{context}
"""

HUMAN_PROMPT = "PREGUNTA: {question}"


def get_llm():
    """Retorna el LLM configurado según LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    # Default: Ollama
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json",
    )


def format_docs(docs) -> str:
    """Formatea los documentos recuperados para incluirlos en el prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", f"Documento {i}")).name
        parts.append(f"[Fuente {i} — {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _safe_parse(parser: PydanticOutputParser, llm_output: str) -> RAGResponse:
    """
    Intenta parsear la salida del LLM. Si falla, construye una respuesta
    de error estructurada en lugar de propagar la excepción.
    """
    try:
        return parser.parse(llm_output)
    except Exception:
        # Intento de extracción manual si el LLM devolvió JSON válido
        try:
            data = json.loads(llm_output)
            return RAGResponse(
                answer=data.get("answer", llm_output),
                sources=data.get("sources", []),
                confidence=data.get("confidence", "baja"),
            )
        except Exception:
            return RAGResponse(
                answer=llm_output,
                sources=[],
                confidence="baja",
            )


def build_rag_chain(retriever):
    """Construye la cadena LCEL: retriever → prompt → LLM → PydanticOutputParser."""
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=RAGResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda text: _safe_parse(parser, text))
    )

    return chain


async def get_rag_response(query: str, retriever, chain=None) -> RAGResponse:
    """
    Función asíncrona principal del sistema RAG.

    Args:
        query: Pregunta del usuario en lenguaje natural.
        retriever: Retriever de ChromaDB configurado con top_k.
        chain: Cadena LCEL pre-construida (opcional, se crea si no se provee).

    Returns:
        RAGResponse con answer, sources y confidence.
    """
    if chain is None:
        chain = build_rag_chain(retriever)
    return await chain.ainvoke(query)


if __name__ == "__main__":
    from ingest import get_embeddings, ingest_documents

    async def demo():
        vectorstore = ingest_documents()
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        chain = build_rag_chain(retriever)

        query = "¿Qué es el mecanismo de atención en los Transformers?"
        print(f"Pregunta: {query}\n")
        response = await get_rag_response(query, retriever, chain)
        print(f"Respuesta: {response.answer}")
        print(f"Confianza: {response.confidence}")
        print(f"Fuentes usadas: {len(response.sources)}")

    asyncio.run(demo())
