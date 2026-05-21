# RAG Simón Bolívar — Agente ReAct + MCP

Sistema de preguntas y respuestas sobre Simón Bolívar construido con RAG (Retrieval-Augmented Generation), un agente ReAct, Model Context Protocol (MCP) y evaluación automática con RAGAS. La interfaz es una aplicación web local con Gradio.

---

## Requisitos previos

- Python 3.10 o superior
- Git
- Conexión a internet (para descargar dependencias y llamar a la API de Anthropic)
- Una API key de [Anthropic](https://console.anthropic.com/)

---

## Instalación y ejecución

### Linux / macOS

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd RAG

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar tu clave:
#   ANTHROPIC_API_KEY=sk-ant-...
```

### Windows

```bat
:: 1. Clonar el repositorio
git clone <url-del-repo>
cd RAG

:: 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate

:: 3. Instalar dependencias
pip install -r requirements.txt

:: 4. Configurar variables de entorno
copy .env.example .env
:: Editar .env con tu editor y agregar tu clave:
::   ANTHROPIC_API_KEY=sk-ant-...
```

### Correr la aplicación

```bash
python app.py
```

Abre el navegador en **http://localhost:7860**.

> La primera ejecución descarga el modelo de embeddings (~90 MB) e indexa los documentos en ChromaDB. Las siguientes arrancan directo.

---

## Estructura del proyecto

```
RAG/
├── app.py                  # Interfaz Gradio (cliente MCP + RAGAS)
├── server.py               # Servidor MCP (ChromaDB + retriever)
├── requirements.txt
├── .env                    # Claves (no se sube al repo)
├── .env.example            # Plantilla de variables
└── chroma_db_bolivar/      # Base vectorial persistente (generada al correr)
```

### Flujo de datos

```
Usuario escribe pregunta
        |
   app.py (Gradio)
        |
   Agente ReAct (Claude)
        |-- decide buscar --> MCP stdio --> server.py
                                               |
                                          ChromaDB
                                          (búsqueda vectorial)
                                               |
                                     top-3 fragmentos relevantes
        |<-- contexto recuperado ---|
        |
   Claude genera respuesta con ese contexto
        |
   Respuesta mostrada en el chat
```

---

## Conceptos teóricos

### RAG (Retrieval-Augmented Generation)

RAG es una arquitectura que mejora las respuestas de un LLM conectándolo a una base de conocimiento propia en lugar de depender solo de lo que aprendió durante el entrenamiento.

El pipeline tiene dos fases:

**Indexación** (ocurre una vez):
1. Los documentos se convierten en vectores numéricos (embeddings)
2. Esos vectores se guardan en una base de datos vectorial (ChromaDB)

**Recuperación** (ocurre en cada pregunta):
1. La pregunta también se convierte en un vector
2. Se buscan los K vectores más similares en la base de datos
3. Los textos originales de esos vectores se pasan al LLM como contexto
4. El LLM genera una respuesta usando ese contexto

La clave es que el LLM nunca recibe vectores — recibe el **texto original** de los documentos más relevantes.

---

### Embeddings

Un embedding es una función que convierte texto en un vector de números reales. Textos con significado similar producen vectores cercanos en el espacio vectorial.

```
"El Libertador"        →  [0.12, -0.45, 0.88, ...]
"Simón Bolívar"        →  [0.11, -0.43, 0.85, ...]  ← cercano
"receta de cocina"     →  [-0.87, 0.22, -0.11, ...]  ← lejano
```

Esta propiedad permite buscar documentos por **significado** en lugar de por coincidencia exacta de palabras. En este proyecto se usa el modelo `all-MiniLM-L6-v2` de `sentence-transformers`, que corre localmente sin necesidad de API.

La similitud entre vectores se mide con **similitud coseno**: dos vectores son más similares cuanto menor es el ángulo entre ellos.

---

### Base de datos vectorial (ChromaDB)

ChromaDB es una base de datos especializada en almacenar vectores y recuperar los más similares a una consulta. A diferencia de una base de datos relacional que busca por coincidencia exacta, ChromaDB encuentra los vecinos más cercanos en el espacio vectorial.

En este proyecto ChromaDB persiste en la carpeta `chroma_db_bolivar/`. Cada entrada almacena el par `(vector, texto_original)`. Cuando se hace una consulta, ChromaDB devuelve el texto original de los K documentos más cercanos al vector de la consulta.

---

### Agente ReAct

ReAct (Reasoning + Acting) es un patrón de agente donde el LLM alterna entre razonar y actuar:

```
1. Thought:  "Necesito buscar información sobre la batalla de Boyacá"
2. Action:   llama a buscar_en_base_vectorial("batalla Boyacá")
3. Observe:  recibe los 3 fragmentos más relevantes de ChromaDB
4. Thought:  "Con esto puedo responder"
5. Answer:   genera la respuesta final
```

La diferencia con RAG puro es que en RAG siempre se busca antes de responder. En ReAct el modelo **decide** cuándo y cómo buscar según la pregunta. Esto le da más flexibilidad: si la pregunta no requiere búsqueda, responde directo.

---

### MCP (Model Context Protocol)

MCP es un protocolo estándar creado por Anthropic que define cómo un LLM se comunica con herramientas externas. Su propósito es similar al de USB: en lugar de que cada herramienta tenga su propia integración ad-hoc con cada modelo, MCP provee una interfaz universal.

**Componentes:**
- **Servidor MCP** (`server.py`): expone herramientas que el LLM puede invocar. En este proyecto expone `buscar_en_base_vectorial`.
- **Cliente MCP** (`app.py`): se conecta al servidor, obtiene la lista de herramientas y se las pasa al agente.
- **Transport stdio**: el cliente lanza el servidor como subproceso y se comunican por stdin/stdout con mensajes JSON-RPC.

La ventaja arquitectónica es que `server.py` puede ser consumido por cualquier cliente MCP (Claude Desktop, otros agentes, otras apps) sin cambiar una línea.

---

### LangSmith (observabilidad)

LangSmith registra cada ejecución del agente como una traza que muestra paso a paso qué hizo el modelo: qué mensajes recibió, qué tools llamó, qué devolvió cada tool, cuánto tardó y cuántos tokens consumió.

Se activa con dos variables en `.env`:
```
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

Las trazas se ven en [smith.langchain.com](https://smith.langchain.com) bajo el proyecto configurado en `LANGCHAIN_PROJECT`.

---

### RAGAS (evaluación del pipeline RAG)

RAGAS es un framework que evalúa automáticamente la calidad de un sistema RAG usando un LLM como juez. Este proyecto usa tres métricas:

**Faithfulness** (fidelidad)
Mide si la respuesta está basada en los contextos recuperados o si el modelo "inventó" información. Un score de 1.0 significa que todo lo afirmado en la respuesta puede encontrarse en los fragmentos recuperados.

**Answer Relevancy** (relevancia de la respuesta)
Mide si la respuesta es pertinente para la pregunta. Funciona generando preguntas artificiales a partir de la respuesta y midiendo su similitud con la pregunta original usando embeddings.

**Context Precision** (precisión del contexto)
Mide si los documentos recuperados son los correctos para responder la pregunta. Compara los contextos recuperados con una respuesta de referencia (`ground_truth`).

Todos los scores van de 0 a 1. Un pipeline bien calibrado debería tener scores superiores a 0.7 en las tres métricas.

---

## Variables de entorno

| Variable | Requerida | Descripción |
|---|---|---|
| `ANTHROPIC_API_KEY` | Sí | Clave de API de Anthropic (chat + evaluación RAGAS) |
| `LANGCHAIN_API_KEY` | No | Clave de LangSmith para trazas de observabilidad |
| `LANGCHAIN_TRACING_V2` | No | `true` para activar trazas en LangSmith |
| `LANGCHAIN_PROJECT` | No | Nombre del proyecto en LangSmith |
