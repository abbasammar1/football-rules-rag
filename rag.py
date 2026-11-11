"""Football Rules RAG: answers questions about the IFAB Laws of the Game 2026/27.

Pipeline: load PDF -> chunk -> embed (bge-base) -> store (ChromaDB)
          -> retrieve top 20 -> rerank (cross-encoder) -> generate (Qwen2.5-3B).
"""

import os
import re
import urllib.request
from pathlib import Path

import chromadb
import torch
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

PDF_URL = "https://downloads.theifab.com/downloads/laws-of-the-game-202627-single-pages?l=en"
PDF_PATH = Path(__file__).parent / "data" / "laws.pdf"
DB_PATH = Path(__file__).parent / ".chroma"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Override with e.g. LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct on a slow CPU machine.
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# bge models expect this prefix on queries (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# If even the best reranked chunk scores below this, refuse instead of guessing.
# Off-topic questions score around -5 to -11; real rules questions score above 0.
MIN_SCORE = 0.0

# What the assistant says when a question isn't about the rules (or the rulebook doesn't cover it).
REFUSAL = "Sorry, I can't help with that. I only answer questions about the football Laws of the Game."

SYSTEM_PROMPT = (
    "You are a football rules assistant. Answer using only the rulebook excerpts provided. "
    "Answer in 1-3 sentences and cite the page, e.g. (p. 145). "
    f"If the excerpts do not contain the answer, reply exactly: {REFUSAL}"
)

LAW_TITLES = {
    1: "The Field of Play", 2: "The Ball", 3: "The Players", 4: "The Players' Equipment",
    5: "The Referee", 6: "The Other Match Officials", 7: "The Duration of the Match",
    8: "The Start and Restart of Play", 9: "The Ball in and out of Play",
    10: "Determining the Outcome of a Match", 11: "Offside", 12: "Fouls and Misconduct",
    13: "Free Kicks", 14: "The Penalty Kick", 15: "The Throw-in", 16: "The Goal Kick",
    17: "The Corner Kick",
}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def normalize(text):
    """Collapse whitespace and straighten curly quotes so text is easy to search."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split())


def ensure_pdf(pdf_path=PDF_PATH):
    """The rulebook isn't stored in the repo; download it from IFAB the first time."""
    if not pdf_path.exists():
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the Laws of the Game from theifab.com ...")
        # The IFAB site rejects Python's default user agent, so send a browser-style one.
        request = urllib.request.Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response:
            pdf_path.write_bytes(response.read())
    return pdf_path


def load_pages(pdf_path=PDF_PATH):
    reader = PdfReader(str(ensure_pdf(pdf_path)))
    return [{"page": i + 1, "text": page.extract_text() or ""} for i, page in enumerate(reader.pages)]


# ---------------------------------------------------------------- chunking

def chunk_fixed(pages, size=200, step=150):
    """Version 1: a new 200-word chunk every 150 words, ignoring the document's structure."""
    chunks = []
    for p in pages:
        words = normalize(p["text"]).split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + size])
            if len(piece) > 100:
                chunks.append({"page": p["page"], "section": "", "text": piece})
    return chunks


def section_label(page_text, previous_label, page_number):
    """Work out which part of the rulebook a page belongs to, from its running header."""
    text = normalize(page_text)
    match = re.search(r"\| Law (\d{1,2}) \|", text)
    if match and int(match.group(1)) in LAW_TITLES:
        n = int(match.group(1))
        return f"Law {n} - {LAW_TITLES[n]}"
    for marker, label in [
        ("| Law changes", "Law changes 2026/27 (what changed this season)"),
        ("| VAR protocol", "VAR protocol"),
        ("| FIFA Quality", "FIFA Quality Programme (ball standards)"),
        ("| Glossary", "Glossary"),
        ("| Practical guidelines", "Practical guidelines for match officials"),
        ("| Notes and modifications", "Notes and modifications (optional protocols, e.g. youth/grassroots)"),
        ("| About the Laws", "About the Laws"),
    ]:
        if marker in text:
            return label
    return previous_label


SUBHEADING = re.compile(r"^\s*(\d{1,2}\.\s+[A-Z][^\n]{2,60})$")


def chunk_by_section(pages, size=120, step=90):
    """Version 2: smaller chunks that never mix two sub-sections, each labelled with its Law
    and sub-heading (e.g. "Law 16 - The Goal Kick | 1. Procedure"), so similar rules from
    different Laws are easier to tell apart."""
    chunks, label, subheading = [], "", ""
    for p in pages:
        new_label = section_label(p["text"], label, p["page"])
        if new_label != label:
            label, subheading = new_label, ""

        # Split the page into blocks at sub-headings such as "1. Procedure".
        blocks, current = [], [subheading, []]
        for line in p["text"].splitlines():
            m = SUBHEADING.match(line)
            if m:
                if current[1]:
                    blocks.append(current)
                current = [normalize(m.group(1)), []]
            else:
                current[1].append(line)
        if current[1]:
            blocks.append(current)
        if blocks:
            subheading = blocks[-1][0]

        for heading, lines in blocks:
            words = normalize(" ".join(lines)).split()
            prefix = " | ".join(x for x in (label, heading) if x)
            for start in range(0, len(words), step):
                piece = " ".join(words[start:start + size])
                if len(piece) > 100:
                    text = f"[{prefix}] {piece}" if prefix else piece
                    chunks.append({"page": p["page"], "section": prefix, "text": text})
    return chunks


CHUNKERS = {"fixed": chunk_fixed, "section": chunk_by_section}


# ---------------------------------------------------------------- the assistant

class RulesAssistant:
    def __init__(self, strategy="section", load_llm=True):
        self.strategy = strategy
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
        self.collection = self._build_index(strategy)
        self.llm = None
        if load_llm:
            from transformers import pipeline
            # Half precision on a GPU / Apple chip; full precision on a plain CPU (e.g. a free HF Space).
            dtype = torch.float32 if DEVICE == "cpu" else torch.float16
            try:
                self.llm = pipeline("text-generation", model=LLM_MODEL, device=DEVICE, dtype=dtype)
            except TypeError:  # older transformers versions call it torch_dtype
                self.llm = pipeline("text-generation", model=LLM_MODEL, device=DEVICE, torch_dtype=dtype)

    def _build_index(self, strategy):
        client = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client.get_or_create_collection(f"laws_{strategy}", metadata={"hnsw:space": "cosine"})
        if collection.count() == 0:
            chunks = CHUNKERS[strategy](load_pages())
            texts = [c["text"] for c in chunks]
            collection.add(
                ids=[str(i) for i in range(len(chunks))],
                documents=texts,
                embeddings=self.embedder.encode(texts, normalize_embeddings=True, batch_size=32).tolist(),
                metadatas=[{"page": c["page"], "section": c["section"]} for c in chunks],
            )
        return collection

    def retrieve(self, question, k=5, candidates=20):
        """Fast embedding search for 20 candidates, then the reranker picks the best k."""
        query = self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=query, n_results=candidates)
        docs, metas = results["documents"][0], results["metadatas"][0]
        scores = self.reranker.predict([(question, d) for d in docs])
        ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)[:k]
        return [{"score": float(s), "text": d, "page": m["page"], "section": m["section"]} for s, d, m in ranked]

    def ask(self, question, k=5):
        hits = self.retrieve(question, k=k)
        if not hits or hits[0]["score"] < MIN_SCORE:
            return {"answer": REFUSAL, "sources": hits}

        context = "\n\n".join(f"[Page {h['page']}] {h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"},
        ]
        out = self.llm(messages, max_new_tokens=160, do_sample=False)
        return {"answer": out[0]["generated_text"][-1]["content"].strip(), "sources": hits}


if __name__ == "__main__":
    import sys
    assistant = RulesAssistant()
    question = " ".join(sys.argv[1:]) or "Can a goalkeeper score directly from a goal kick?"
    result = assistant.ask(question)
    print(result["answer"])
    print("Sources:", [h["page"] for h in result["sources"]])
