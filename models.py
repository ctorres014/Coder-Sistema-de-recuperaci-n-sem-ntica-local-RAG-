from typing import List
from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline."""

    answer: str = Field(
        description="La respuesta a la pregunta del usuario basada únicamente en el contexto proporcionado"
    )
    sources: List[str] = Field(
        description="Lista de fragmentos del contexto utilizados para generar la respuesta"
    )
    confidence: str = Field(
        description="Nivel de confianza en la respuesta: 'alta', 'media', 'baja', o 'sin_informacion'"
    )
