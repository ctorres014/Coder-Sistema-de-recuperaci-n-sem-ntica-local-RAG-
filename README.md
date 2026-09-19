# Pre-entrega 3: Sistema RAG Local con ChromaDB y Ollama

Sistema de recuperación semántica (RAG) End-to-End implementado con LangChain, ChromaDB y Ollama (LLM local).

## Arquitectura

```
data/ (documentos .txt/.md)
    ↓  RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
    ↓  Embeddings (nomic-embed-text via Ollama)
    ↓  ChromaDB (persistente en ./vectorstore)
                    ↓
              [Query del usuario]
                    ↓
         Búsqueda de similitud (top-k=4)
                    ↓
         Prompt con contexto + pregunta
                    ↓
         LLM (llama3.2 via Ollama) — asíncrono
                    ↓
         PydanticOutputParser → RAGResponse
```

## Estructura del Proyecto

```
Entrega3/
├── data/                        # Documentos fuente
│   ├── introduccion_ml.txt
│   ├── redes_neuronales.txt
│   ├── procesamiento_lenguaje.txt
│   └── vectores_embeddings.txt
├── notebooks/
│   └── rag_demo.ipynb           # Demo interactiva
├── ingest.py                    # Módulo de ingesta (chunking + ChromaDB)
├── rag_chain.py                 # Cadena RAG asíncrona (LCEL)
├── models.py                    # Modelo Pydantic de respuesta
├── main.py                      # Script de demostración con 2 pruebas
├── ask.py                       # CLI para hacer preguntas individuales
├── requirements.txt
├── .env.example
└── .gitignore
```

## Requisitos Previos

- **Python 3.10–3.13** — Python 3.14 no es compatible con chromadb todavía. Verificá tu versión con `python3 --version`.
- **Ollama** instalado y corriendo: [https://ollama.ai](https://ollama.ai)

---

## Instalación

### macOS

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Entrega3

# 2. Verificar versión de Python (debe ser 3.10–3.13)
python3 --version

# Si la versión es 3.14, usar python3.13 explícitamente:
# python3.13 -m venv venv
# Si la versión es 3.10–3.13, usar:
python3 -m venv venv

# 3. Activar el entorno virtual
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Descargar modelos en Ollama
ollama pull llama3.2
ollama pull nomic-embed-text

# 6. Configurar variables de entorno
cp .env.example .env
```

### Linux

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Entrega3

# 2. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelos en Ollama
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. Configurar variables de entorno
cp .env.example .env
```

### Windows (PowerShell)

```powershell
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Entrega3

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelos en Ollama
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. Configurar variables de entorno
copy .env.example .env
```

> **Windows (CMD):** reemplazar el paso 3 por `venv\Scripts\activate.bat`

---

## Configuración

Editá el archivo `.env` si querés cambiar el proveedor o modelo:

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` o `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `OLLAMA_MODEL` | `llama3.2` | Modelo de chat Ollama |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Modelo de embeddings Ollama |
| `OPENAI_API_KEY` | — | API key de OpenAI (si `LLM_PROVIDER=openai`) |

---

## Paso 1: Ejecutar la Ingesta

La ingesta lee los archivos de `data/`, los divide en chunks de 500 tokens y los persiste en ChromaDB. **Solo es necesario correrla una vez**; las siguientes ejecuciones detectan el vectorstore existente y lo reutilizan.

**macOS / Linux:**
```bash
python3 ingest.py
```

**Windows:**
```cmd
python ingest.py
```

Salida esperada:
```
=== Iniciando ingesta de documentos ===
  Documentos cargados: 4
  Chunks generados: 24 (chunk_size=500 tokens, overlap=50)
  Generando embeddings y persistiendo en ChromaDB...
  Vectorstore creado en './vectorstore'
=== Ingesta completada ===
```

Si el vectorstore ya existe, la salida será:
```
Vectorstore existente detectado. Cargando sin re-indexar.
```

Para forzar una re-ingesta completa (por ejemplo, si agregaste o modificaste documentos en `data/`):

**macOS / Linux:**
```bash
python3 ingest.py --force
```

**Windows:**
```cmd
python ingest.py --force
```

---

## Paso 2: Hacer una Pregunta

`ask.py` es el script de entrada al sistema RAG. Internamente llama a la función asíncrona `get_rag_response(query)` definida en `rag_chain.py`, que busca en ChromaDB, construye el prompt con el contexto y devuelve un objeto `RAGResponse` parseado por `PydanticOutputParser`.

### Opción A — Una pregunta por argumento (recomendado)

**macOS / Linux:**
```bash
python3 ask.py "¿Qué es el mecanismo de atención en los Transformers?"
```

**Windows:**
```cmd
python ask.py "¿Qué es el mecanismo de atención en los Transformers?"
```

Salida esperada:
```
Pregunta : ¿Qué es el mecanismo de atención en los Transformers?
Buscando en documentos...

============================================================
Confianza : alta

Respuesta :
El mecanismo de atención permite al modelo enfocarse en partes
relevantes de la secuencia de entrada al generar cada token de
salida. Se calcula como: Attention(Q,K,V) = softmax(QK^T / √d_k) * V

Fragmentos usados (1):
  [1] Self-Attention: cada posición atiende a todas las demás...
============================================================
```

### Opción B — Modo interactivo

Sin argumento, el script pide la pregunta por teclado.

**macOS / Linux:**
```bash
python3 ask.py
```

**Windows:**
```cmd
python ask.py
```

```
Ingresá tu pregunta: ¿Qué es el overfitting?
```

### Opción C — Demo completa con 2 pruebas automáticas

Ejecuta dos pruebas que verifican los criterios de aceptación del sistema:
- **Prueba 1:** pregunta cuya respuesta **está** en los documentos → confianza `alta`
- **Prueba 2:** pregunta trampa cuya respuesta **no está** → el modelo responde que no tiene esa información (no alucina)

**macOS / Linux:**
```bash
python3 main.py
```

**Windows:**
```cmd
python main.py
```

### Opción D — Notebook interactivo

**macOS / Linux:**
```bash
jupyter notebook notebooks/rag_demo.ipynb
```

**Windows:**
```cmd
jupyter notebook notebooks\rag_demo.ipynb
```

---

## Componentes Técnicos

### Módulo de Ingesta (`ingest.py`)

- `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `chunk_size=500` tokens y `chunk_overlap=50` tokens
- Verificación de existencia del vectorstore antes de re-indexar (optimización de costo)
- Soporta `.txt` y `.md`

### Cadena RAG (`rag_chain.py`)

Implementa la función asíncrona principal del sistema:

```python
async def get_rag_response(query: str, retriever, chain=None) -> RAGResponse:
    ...
```

Internamente:
1. Busca similitud en ChromaDB con el embedding de `query`
2. Formatea los fragmentos recuperados con su fuente de origen
3. Llama al LLM de forma asíncrona (`chain.ainvoke`)
4. Parsea la respuesta con `PydanticOutputParser` → `RAGResponse`

La cadena LCEL tiene esta forma:
```
{retriever | format_docs, question} | prompt | llm | StrOutputParser | PydanticOutputParser
```

El prompt instruye al modelo a responder `"sin_informacion"` si la respuesta no está en el contexto. El `top_k=4` evita el problema "Lost in the Middle".

### Modelo de Salida (`models.py`)

```python
class RAGResponse(BaseModel):
    answer: str          # Respuesta basada en el contexto
    sources: List[str]   # Fragmentos del contexto utilizados (referencias)
    confidence: str      # "alta" | "media" | "baja" | "sin_informacion"
```

---

## Dataset de Ejemplo

4 documentos sobre Inteligencia Artificial y Machine Learning:

| Archivo | Contenido |
|---|---|
| `introduccion_ml.txt` | Tipos de aprendizaje, algoritmos clásicos, evaluación |
| `redes_neuronales.txt` | Arquitecturas (CNN, RNN, Transformer), entrenamiento |
| `procesamiento_lenguaje.txt` | NLP, embeddings, LLMs, fine-tuning |
| `vectores_embeddings.txt` | Embeddings, bases de datos vectoriales, RAG |

**Pregunta válida:** `"¿Qué es el mecanismo de atención en los Transformers?"`

**Pregunta trampa:** `"¿Cuál es la receta de la paella valenciana?"` → el modelo responde que no tiene esa información.

---

## Errores Comunes Evitados

- **Embeddings no coincidentes**: se usa el mismo modelo (`nomic-embed-text`) en ingesta y consulta
- **Contexto infinito**: `top_k=4` (máximo recomendado: 3-5)
- **Re-indexado innecesario**: verificación de existencia del vectorstore al inicio
- **API keys en código**: todo via variables de entorno en `.env`
