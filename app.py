"""
app.py — Interfaz Gradio para RAG de Simón Bolívar.
Correr: python app.py  →  http://localhost:7860
"""
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv(Path(__file__).parent / ".env")

SERVER_PATH = str(Path(__file__).parent / "server.py")

_mcp_client = None
_agent = None
_tools = None

EJEMPLOS = [
    "¿Quién fue Simón Bolívar?",
    "¿Qué fue la batalla de Boyacá?",
    "¿Qué es la Gran Colombia?",
    "¿Dónde y cuándo murió Bolívar?",
]


async def startup():
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
    print(">>> Servidor MCP listo. Puedes hacer preguntas.")


async def responder(message: str, history: list):
    if not message.strip():
        return "", history
    if _agent is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "El sistema aún está iniciando, espera unos segundos e intenta de nuevo."})
        return "", history
    resultado = await _agent.ainvoke({"messages": [HumanMessage(content=message)]})
    respuesta = resultado["messages"][-1].content
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": respuesta})
    return "", history


async def usar_ejemplo(ejemplo: str, history: list):
    return await responder(ejemplo, history)


with gr.Blocks(
    title="RAG Simón Bolívar",
    theme=gr.themes.Soft(),
    css="""
        #titulo { text-align: center; margin-bottom: 4px; }
        #subtitulo { text-align: center; color: #666; margin-bottom: 20px; }
        #chatbot { border-radius: 12px; }
        #send-btn { min-width: 100px; }
        .ejemplo-btn { border-radius: 20px !important; font-size: 13px !important; }
    """,
) as demo:

    gr.Markdown("# Simón Bolívar — Asistente RAG", elem_id="titulo")
    gr.Markdown(
        "Agente ReAct con recuperación vectorial via MCP · Base de conocimiento en ChromaDB",
        elem_id="subtitulo",
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=440,
        show_label=False,
        avatar_images=(None, "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Bolivar_by_Roulin.jpg/220px-Bolivar_by_Roulin.jpg"),
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Escribe tu pregunta sobre Simón Bolívar…",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Enviar", variant="primary", scale=1, elem_id="send-btn")

    with gr.Row():
        for ejemplo in EJEMPLOS:
            gr.Button(ejemplo, size="sm", elem_classes="ejemplo-btn").click(
                fn=usar_ejemplo,
                inputs=[gr.Textbox(value=ejemplo, visible=False), chatbot],
                outputs=[msg_input, chatbot],
            )

    clear_btn = gr.ClearButton([msg_input, chatbot], value="Limpiar conversación", size="sm")

    msg_input.submit(responder, [msg_input, chatbot], [msg_input, chatbot])
    send_btn.click(responder, [msg_input, chatbot], [msg_input, chatbot])

    demo.load(startup)


if __name__ == "__main__":
    demo.launch()
