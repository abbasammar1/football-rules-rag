"""Ref AI: a chat app for the football rules assistant, styled after the Figma design.

Run:  python app.py            (full assistant; SHARE=1 for a temporary public link)
      PREVIEW=1 python app.py  (no LLM; shows the best-matching rule instead)
"""

import html
import os

import gradio as gr
from markdown_it import MarkdownIt

from rag import REFUSAL, RulesAssistant

PREVIEW = os.environ.get("PREVIEW") == "1"
assistant = RulesAssistant(strategy="section", load_llm=not PREVIEW)

CARDS = [
    ("01", "Throw-in into your own goal?", "What the referee gives.",
     "A throw-in is thrown directly into the thrower's own goal. What is awarded?"),
    ("02", "Offside from a throw-in?", "The restart that cancels offside.",
     "Is there an offside offence if a player receives the ball directly from a throw-in?"),
    ("03", "The 8-second rule", "When the keeper holds on too long.",
     "What happens if the goalkeeper holds the ball for more than eight seconds?"),
]


MD = MarkdownIt("commonmark", {"html": False})  # LLM text is rendered as markdown, raw HTML disabled


def get_answer(question):
    """Returns (answer_text, sources) where sources is a list of "p. 145 · Law 16 - The Goal Kick"."""
    if PREVIEW:
        hits = assistant.retrieve(question)
        if not hits or hits[0]["score"] < 0:
            return REFUSAL, []
        best = hits[0]
        text = best["text"].split("] ", 1)[-1]
        return f"*Preview mode: the best-matching rule.*\n\n> {text[:420]}…", [f"p. {best['page']} · {best['section']}"]

    result = assistant.ask(question)
    if not result["sources"] or result["answer"].startswith("Sorry, I can't help"):
        return result["answer"], []
    seen, refs = set(), []
    for h in result["sources"][:3]:
        if h["page"] not in seen:
            seen.add(h["page"])
            refs.append(f"p. {h['page']} · {h['section'] or 'Laws of the Game'}")
    return result["answer"], refs


def render(turns):
    """Draw the conversation: a "YOU" bubble on the right, then Ref's answer as plain text."""
    if not turns:
        return ""
    title = turns[0]["q"]
    if len(title) > 60:
        title = title[:60].rsplit(" ", 1)[0] + "…"
    parts = [f"<div class='convo-title'>{html.escape(title)}</div>"]
    for t in turns:
        parts.append(f"<div class='you-label'>YOU</div><div class='you'>{html.escape(t['q'])}</div>")
        if t["a"] is None:
            body = "<p class='thinking'><span class='dot'></span>Checking the rulebook…</p>"
        else:
            body = MD.render(t["a"])
            if t["refs"]:
                body += "<div class='refs'>" + "".join(f"<span>{html.escape(r)}</span>" for r in t["refs"]) + "</div>"
        parts.append("<div class='ref-head'><span class='avatar'>R</span><span class='ref-name'>Ref</span></div>"
                     f"<div class='ref-body'>{body}</div>")
    return "<div class='convo'>" + "".join(parts) + "</div>"


def send(message, turns):
    """Outputs: textbox, conversation html, turns state, inspiration cards, hero."""
    message = (message or "").strip()
    if not message:
        yield gr.update(), gr.update(), turns, gr.update(), gr.update()
        return
    turns = (turns or []) + [{"q": message, "a": None, "refs": []}]
    hide = gr.update(visible=False)
    follow_up = gr.update(value="", placeholder="Ask a follow-up…")
    yield follow_up, gr.update(value=render(turns), visible=True), turns, hide, hide
    turns[-1]["a"], turns[-1]["refs"] = get_answer(message)
    yield follow_up, gr.update(value=render(turns), visible=True), turns, hide, hide


def new_chat():
    show = gr.update(visible=True)
    box = gr.update(value="", placeholder="Ask a rules question, e.g. is it offside from a throw-in?")
    return box, gr.update(value="", visible=False), [], show, show


CSS = """
:root, .dark {
  --bg: #0C0D0C; --panel: #111210; --surface: #181C14; --raised: #22251E;
  --line: #292C26; --line-strong: #343D27; --text: #ECEEE6; --muted: #95998F;
  --lime: #D5F581; --lime-ink: #18220C;
}
body, .gradio-container, .main, .wrap, .contain { background: var(--bg) !important; }
.gradio-container, .gradio-container > .main, .gradio-container .wrap, .fillable { max-width: 100% !important; }
.gradio-container { padding: 0 !important; margin: 0 !important;
  font-family: 'Inter', system-ui, sans-serif !important; color: var(--text) !important; }
footer { display: none !important; }
.app, main, .fillable, .gradio-container > .main, .gradio-container .wrap { padding: 0 !important; margin: 0 !important; }
* { --body-text-color: #ECEEE6; --block-background-fill: transparent; --block-border-width: 0px;
    --input-background-fill: transparent; --border-color-primary: #292C26; }

/* layout */
#shell { gap: 0 !important; min-height: 100vh; flex-wrap: nowrap !important; margin: 0 !important; }
#sidebar { background: var(--panel); border-right: 1px solid var(--line); padding: 28px 22px !important; min-height: 100vh;
  min-width: 260px !important; max-width: 260px !important; gap: 10px !important; }
#main { padding: 0 !important; gap: 0 !important; }

/* sidebar */
.brand { font-size: 30px; font-weight: 700; letter-spacing: -0.04em; color: var(--text); margin-bottom: 18px; }
.brand sup { font-size: 10px; color: var(--lime); font-weight: 600; margin-left: 3px; letter-spacing: 0; }
#new-chat { background: var(--lime) !important; color: var(--lime-ink) !important; border: 0 !important;
  border-radius: 8px !important; font-weight: 600 !important; font-size: 14px !important;
  justify-content: flex-start !important; padding: 12px 16px !important; box-shadow: none !important; }
#new-chat:hover { filter: brightness(1.05); }
.side-label { font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em; color: var(--muted);
  margin: 22px 0 8px; text-transform: uppercase; }
.side-item { font-size: 13.5px; color: var(--text); padding: 7px 0; line-height: 1.35; }
.side-item small { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
.side-item a { color: var(--text); text-decoration: none; border-bottom: 1px solid var(--line-strong); }
.stack-row { display: flex; justify-content: space-between; font-size: 12.5px; padding: 6px 0;
  border-bottom: 1px solid var(--line); color: var(--muted); }
.stack-row b { color: var(--text); font-weight: 500; }
.side-card { margin-top: 26px; border: 1px solid var(--line-strong); background: var(--surface);
  border-radius: 8px; padding: 14px 16px; font-size: 13px; }
.side-card a { color: var(--lime); text-decoration: none; }
.side-card span { color: var(--muted); display: block; margin-top: 4px; }

/* top bar */
.topbar { display: flex; align-items: center; gap: 12px; padding: 20px 64px;
  border-bottom: 1px solid var(--line); font-size: 15px; font-weight: 600; color: var(--text); }
.badge { background: var(--lime-ink); color: var(--lime); font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.06em; padding: 3px 7px; border-radius: 4px; }
.topbar .right { margin-left: auto; font-size: 12.5px; font-weight: 400; color: var(--muted); }

/* hero */
#content { max-width: 1000px; width: 100%; margin: 0 !important; padding: 0 64px !important; }
.hero { padding-top: 7vh; }
#content .html-container, #content .prose { padding-left: 0 !important; padding-right: 0 !important; }
.eyebrow { color: var(--lime); font-size: 12px; font-weight: 500; letter-spacing: 0.04em; margin-bottom: 26px; }
.hero h1 { font-size: clamp(52px, 7.5vw, 96px) !important; font-weight: 700 !important; line-height: 1.02 !important;
  letter-spacing: -0.05em !important; color: var(--text) !important; margin: 0 0 22px !important; }
.hero p { color: var(--muted); font-size: 17px; margin: 0 0 40px; }

/* conversation (Figma reply design) */
#convo { margin-top: 36px; }
.convo-title { font-size: 22px; font-weight: 600; color: var(--text); margin-bottom: 36px; letter-spacing: -0.01em; }
.you-label { text-align: right; color: var(--muted); font-size: 11px; letter-spacing: 0.08em; margin-bottom: 10px; }
.you { margin-left: auto; max-width: 78%; width: fit-content; background: var(--raised); color: var(--text);
  border-radius: 14px; padding: 14px 20px; font-size: 16px; line-height: 1.6; margin-bottom: 40px; }
.ref-head { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
.avatar { width: 32px; height: 32px; border-radius: 50%; background: var(--lime); color: var(--lime-ink);
  display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 15px; }
.ref-name { color: var(--lime); font-weight: 600; font-size: 15px; }
.ref-body { padding-left: 46px; color: var(--text); font-size: 16px; line-height: 1.7; margin-bottom: 44px; }
.ref-body p { margin: 0 0 14px; color: var(--text); }
.ref-body h1, .ref-body h2, .ref-body h3 { color: var(--text); font-weight: 600; letter-spacing: -0.01em; margin: 22px 0 10px; }
.ref-body strong { color: var(--text); font-weight: 600; }
.ref-body ul, .ref-body ol { margin: 0 0 14px 20px; }
.ref-body blockquote { border-left: 2px solid var(--lime); margin: 0 0 14px; padding-left: 14px; color: var(--muted); }
.ref-body em { color: var(--muted); }
.refs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.refs span { font-size: 12px; color: var(--muted); border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; }
.thinking { color: var(--muted) !important; display: flex; align-items: center; gap: 10px; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--lime); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: .25 } 50% { opacity: 1 } }

/* composer */
#composer { background: var(--surface); border: 1px solid var(--line-strong); border-radius: 8px;
  padding: 14px 14px 14px 18px !important; align-items: flex-end !important; gap: 12px !important; margin-top: 8px; }
#composer textarea { background: transparent !important; border: 0 !important; box-shadow: none !important;
  color: var(--text) !important; font-size: 16px !important; padding: 6px 0 !important; min-height: 64px; }
#composer textarea::placeholder { color: var(--muted) !important; }
/* no blue focus ring: just a soft lime edge on the whole box while typing */
#composer textarea:focus, #composer textarea:focus-visible, #composer *:focus { outline: none !important; box-shadow: none !important; border-color: transparent !important; }
#composer .wrap, #composer .container, #composer label, #composer .input-container { border: 0 !important; box-shadow: none !important; }
#composer:focus-within { border-color: #5a6b3a !important; }
* { --input-border-color-focus: transparent; --input-shadow-focus: none; --color-accent: #D5F581; --color-accent-soft: transparent; }
#composer .block, #composer label { background: transparent !important; border: 0 !important; box-shadow: none !important; }
#send { background: var(--lime) !important; color: var(--lime-ink) !important; border: 0 !important;
  border-radius: 8px !important; min-width: 44px !important; max-width: 44px !important; height: 44px !important;
  font-size: 18px !important; font-weight: 700 !important; padding: 0 !important; }

/* inspiration cards */
.cards-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 13px; margin: 44px 0 14px; }
#cards { gap: 14px !important; }
.card { background: transparent !important; border: 1px solid var(--line) !important; border-radius: 8px !important;
  color: var(--text) !important; text-align: left !important; justify-content: flex-start !important;
  align-items: flex-start !important; white-space: pre-line !important; padding: 16px 18px !important;
  font-size: 15px !important; font-weight: 600 !important; line-height: 1.5 !important; min-height: 104px; display: block !important;
  box-shadow: none !important; }
.card:hover { border-color: var(--lime) !important; background: var(--surface) !important; }
.card::first-line { color: var(--muted); font-size: 11px; font-weight: 400; line-height: 2.6; }

.disclaimer { text-align: center; color: var(--muted); font-size: 11.5px; padding: 48px 0 28px; }
"""

HEAD = '<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'

with gr.Blocks(title="Ref AI · Football Rules Assistant", fill_height=True, fill_width=True) as demo:
    with gr.Row(elem_id="shell", equal_height=False):
        with gr.Column(elem_id="sidebar", scale=0):
            gr.HTML("<div class='brand'>ref<sup>AI</sup></div>")
            new_btn = gr.Button("＋  New chat", elem_id="new-chat")
            gr.HTML(
                "<div class='side-label'>The rulebook</div>"
                "<div class='side-item'><a href='https://www.theifab.com/laws-of-the-game-documents/' target='_blank'>"
                "IFAB Laws of the Game</a><small>2026/27 edition · 17 Laws · 260 pages</small></div>"
                "<div class='side-label'>Tips</div>"
                "<div class='side-item'>Describe a real situation<small>e.g. the ball hits the referee and goes in</small></div>"
                "<div class='side-item'>Check the page<small>every answer shows where it came from</small></div>"
                "<div class='side-item'>Rules only<small>it won't guess about players, teams or results</small></div>"
                "<div class='side-card'>Built by Abbas Ammar"
                "<span><a href='https://github.com/abbasammar1' target='_blank'>GitHub ↗</a></span></div>"
            )

        with gr.Column(elem_id="main", scale=1):
            gr.HTML("<div class='topbar'>Laws of the Game <span class='badge'>2026/27</span>"
                    "<span class='right'>Answers only from the official rulebook</span></div>")
            with gr.Column(elem_id="content"):
                with gr.Column(visible=True) as hero:
                    gr.HTML("<div class='hero'><div class='eyebrow'>IFAB LAWS OF THE GAME 2026/27</div>"
                            "<h1>Know the law.<br>Make the call.</h1>"
                            "<p>Ask any rules question. Every answer cites the page it came from.</p></div>")
                convo = gr.HTML(elem_id="convo", visible=False)
                turns = gr.State([])
                with gr.Row(elem_id="composer"):
                    box = gr.Textbox(placeholder="Ask a rules question, e.g. is it offside from a throw-in?",
                                     show_label=False, container=False, lines=2, max_lines=6, scale=1)
                    send_btn = gr.Button("↑", elem_id="send", scale=0)
                with gr.Column(visible=True) as inspiration:
                    gr.HTML("<div class='cards-head'><span>Try a real situation</span><span>01 / 03</span></div>")
                    with gr.Row(elem_id="cards"):
                        card_buttons = [gr.Button(f"{n}\n{title}", elem_classes="card") for n, title, sub, _ in CARDS]
                gr.HTML("<div class='disclaimer'>AI can make mistakes. Check the page it cites.</div>")

    outputs = [box, convo, turns, inspiration, hero]
    box.submit(send, [box, turns], outputs)
    send_btn.click(send, [box, turns], outputs)
    for btn, (_, _, _, question) in zip(card_buttons, CARDS):
        btn.click(lambda t, q=question: (yield from send(q, t)), [turns], outputs)
    new_btn.click(new_chat, None, outputs)

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get('PORT', 7860)), share=os.environ.get('SHARE') == '1', css=CSS, head=HEAD, theme=gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]))
