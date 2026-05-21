"""
server.py — Servidor MCP (stdio) para RAG de Simón Bolívar.
Expone: buscar_en_base_vectorial(query: str) -> str
Transport: stdio — el notebook lo lanza como subproceso via MultiServerMCPClient.
"""
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).parent.resolve()
load_dotenv(_HERE / ".env", override=False)

# Documentos de Bolívar 
DOCUMENTS = [
    "Simón Bolívar nació el 24 de julio de 1783 en Caracas, entonces parte del Virreinato de Nueva Granada.",
    "Es conocido como El Libertador por su papel militar y político en la independencia de Venezuela, Colombia, Ecuador, Perú y Bolivia frente al imperio español.",
    "En 1819 lideró la campaña que culminó en la batalla de Boyacá, decisiva para la independencia de la Nueva Granada.",
    "En 1819 pronunció el Discurso de Angostura, donde expuso ideas sobre república y gobierno para la América hispana.",
    "Fue presidente de Bolivia en 1825 y propulsó la creación de la República de Bolivia como estado soberano.",
    "Presidió la Gran Colombia, estado que unió gran parte del norte de Sudamérica entre 1819 y 1831.",
    "Murió el 17 de diciembre de 1830 en Santa Marta (actual Colombia); su legado simboliza la unidad latinoamericana.",
]

CHROMA_DIR = str(_HERE / "chroma_db_bolivar")
EMBED_MODEL = "all-MiniLM-L6-v2"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
_persist = Path(CHROMA_DIR)

if _persist.exists() and any(_persist.iterdir()):
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_fn)
    if len(vectorstore.get().get("ids", [])) == 0:
        vectorstore = Chroma.from_documents(
            [Document(page_content=t) for t in DOCUMENTS],
            embedding_fn,
            persist_directory=CHROMA_DIR,
        )
else:
    _persist.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        [Document(page_content=t) for t in DOCUMENTS],
        embedding_fn,
        persist_directory=CHROMA_DIR,
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BolívarRAG")


@mcp.tool()
def buscar_en_base_vectorial(query: str) -> str:
    """Busca hechos sobre la vida, batallas y legado de Simón Bolívar en la base vectorial."""
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontraron documentos relevantes."
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


if __name__ == "__main__":
    mcp.run(transport="stdio")
