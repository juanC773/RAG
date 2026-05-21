"""
app.py — Interfaz Gradio para RAG de Simón Bolívar.
Tabs: Chat (agente ReAct + MCP) | Evaluación RAGAS
Correr: python app.py  →  http://localhost:7860
"""
import sys
from pathlib import Path

import gradio as gr
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

load_dotenv(Path(__file__).parent / ".env")

SERVER_PATH = str(Path(__file__).parent / "server.py")

PREGUNTAS_EVAL = [
    "¿Quién es Simón Bolívar?",
    "¿Qué importancia tuvo la batalla de Boyacá en la carrera de Simón Bolívar?",
]

GROUND_TRUTH = [
    "Simón Bolívar fue un militar y político venezolano nacido en Caracas en 1783, conocido como El Libertador por liderar la independencia de varias naciones sudamericanas del dominio español.",
    "La batalla de Boyacá en 1819 fue decisiva para la independencia de la Nueva Granada y consolidó la posición militar de Bolívar en la campaña libertadora.",
]

# Estado global: cliente MCP y agente se crean una vez al arrancar la app
_mcp_client = None
_agent = None
_tools = None


async def startup():
    """Inicializa el cliente MCP y el agente ReAct una sola vez."""
    global _mcp_client, _agent, _tools
    _mcp_client = MultiServerMCPClient({
        "bolivar_rag": {
            "command": sys.executable,
            "args": [SERVER_PATH],
            "transport": "stdio",
        }
    })
    _tools = await _mcp_client.get_tools()
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    _agent = create_react_agent(llm, _tools)


async def ejecutar_agente_mcp(pregunta: str) -> str:
    resultado = await _agent.ainvoke({"messages": [HumanMessage(content=pregunta)]})
    return resultado["messages"][-1].content


async def get_contexts_mcp(query: str) -> list:
    """Invoca la tool MCP directamente (sin agente) para obtener contextos para RAGAS."""
    tool_buscar = next(t for t in _tools if t.name == "buscar_en_base_vectorial")
    raw = await tool_buscar.ainvoke({"query": query})
    contextos = [
        line.split("] ", 1)[1].strip()
        for line in raw.split("\n\n") if "] " in line
    ]
    return contextos if contextos else [raw]


async def responder(message: str, history: list):
    if not message.strip():
        return "", history
    respuesta = await ejecutar_agente_mcp(message)
    history.append((message, respuesta))
    return "", history


async def run_ragas_evaluation():
    registros = []
    for q in PREGUNTAS_EVAL:
        ctx = await get_contexts_mcp(q)
        resp = await ejecutar_agente_mcp(q)
        registros.append({"question": q, "answer": resp, "contexts": ctx})

    eval_ds = Dataset.from_dict({
        "question": [r["question"] for r in registros],
        "answer":   [r["answer"]   for r in registros],
        "contexts": [r["contexts"] for r in registros],
        "ground_truth": GROUND_TRUTH,
    })

    ragas_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0))
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    scores = evaluate(
        eval_ds,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )
    df = scores.to_pandas()[
        ["user_input", "faithfulness", "answer_relevancy", "context_precision"]
    ]
    return df, "Evaluación completada"


with gr.Blocks(title="RAG Simón Bolívar") as demo:
    gr.Markdown("# RAG Simón Bolívar — Agente ReAct + MCP")

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=420)
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Pregunta sobre Simón Bolívar…",
                label="",
                scale=8,
            )
            send_btn = gr.Button("Enviar", variant="primary", scale=1)
        clear_btn = gr.ClearButton([msg_input, chatbot], value="Limpiar")

        msg_input.submit(responder, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(responder, [msg_input, chatbot], [msg_input, chatbot])

    with gr.Tab("Evaluación RAGAS"):
        gr.Markdown(
            "Evalúa el pipeline con **faithfulness**, **answer relevancy** y **context precision**."
        )
        eval_btn = gr.Button("Ejecutar Evaluación", variant="primary")
        status        = gr.Textbox(label="Estado", interactive=False)
        results_table = gr.Dataframe(label="Resultados")

        eval_btn.click(run_ragas_evaluation, outputs=[results_table, status])

    # Inicializa el servidor MCP al cargar la app (una sola vez)
    demo.load(startup)


if __name__ == "__main__":
    demo.launch()
