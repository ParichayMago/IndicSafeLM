"""Minimal chatbot on the IndicSafeLM model.

It answers everyday requests and refuses harmful ones. Each reply carries a
short verdict line so the intended behaviour is visible at a glance.

Run: ``python scripts/app.py`` then open http://127.0.0.1:7862
"""

from __future__ import annotations

from .assistant import Assistant, examples

VERDICT = {
    "helpful": "✓ answered",
    "declined": "⛔ refused — harmful request",
    "replaced": "⛔ refused — harmful request (safety check)",
    "error": "⚠ no reply — please try again",
}

CSS = """
.gradio-container { max-width: 720px !important; margin: 0 auto !important;
  font-family: "Noto Sans Devanagari", "Kohinoor Devanagari", "Noto Sans Bengali", "Kohinoor Bangla",
  "Noto Sans Tamil", "Tamil Sangam MN", system-ui, sans-serif !important; }
#title h1 { font-size: 1.8rem; margin: 16px 0 2px; }
#title p { margin: 0 0 6px; opacity: .7; }
#chat .message { font-size: 1.15rem !important; line-height: 1.6 !important; }
footer { display: none !important; }
"""


def build(assistant: Assistant | None = None):
    import gradio as gr

    assistant = assistant or Assistant()
    hi, bn, ta = examples("hi"), examples("bn"), examples("ta")
    # Order in examples(): reminder, thank-you, birthday, notebook, polite rewrite, insult, threat
    helpful = [hi[0], hi[1], bn[2], ta[0]]
    harmful = [hi[5], hi[6], bn[5], ta[5]]

    def chat(message, history):
        message = (message or "").strip()
        if not message:
            return "", history
        reply = assistant.respond(message)
        text = reply.text or reply.note
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"{text}\n\n*{VERDICT[reply.kind]}*"},
        ]
        return "", history

    with gr.Blocks(title="IndicSafeLM chatbot") as app:
        gr.HTML('<div id="title"><h1>IndicSafeLM</h1>'
                "<p>A Hindi · Bengali · Tamil chatbot that helps with everyday requests and refuses harmful ones.</p></div>")
        bot = gr.Chatbot(height=430, elem_id="chat", show_label=False)
        box = gr.Textbox(placeholder="Type in Hindi, Bengali or Tamil and press Enter…", show_label=False, autofocus=True)
        with gr.Row():
            clear = gr.Button("Clear chat", size="sm")
        gr.Examples(helpful, inputs=box, label="Everyday requests (should answer)")
        gr.Examples(harmful, inputs=box, label="Harmful requests (should refuse)")
        box.submit(chat, [box, bot], [box, bot], concurrency_limit=1)
        clear.click(lambda: ("", []), None, [box, bot])
    return app


def main(port: int = 7862) -> None:
    import gradio as gr

    build().queue().launch(server_name="127.0.0.1", server_port=port, css=CSS, theme=gr.themes.Default())
