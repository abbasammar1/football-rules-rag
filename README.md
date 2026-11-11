# ⚽ Ref AI: a football rules assistant

Ask a question about the football **Laws of the Game 2026/27** and get an answer taken only from the official IFAB rulebook,
with the page it came from. If the rulebook doesn't cover it ("Who won the 2022 World Cup?"), it politely says it only answers
questions about the Laws of the Game, instead of guessing.


Built with open-source models only (no paid APIs), using retrieval-augmented generation (RAG).

## How it works

```
Rulebook PDF (260 pages)
   │  1. Load and chunk: cut into ~120-word chunks, each labelled with its Law and sub-section
   ▼        e.g. "[Law 16 - The Goal Kick | 1. Procedure] The ball must be stationary..."
Embeddings (BAAI/bge-base-en-v1.5)  ──►  ChromaDB vector store
   │  2. Retrieve: embed the question, fetch the 20 closest chunks
   ▼
Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
   │  3. Rerank: read each (question, chunk) pair and keep the best 5.
   │     If even the best one scores below 0, politely refuse (no LLM call).
   ▼
LLM (Qwen2.5-3B-Instruct)
      4. Generate: answer in 1–3 sentences using only those 5 chunks, citing the page
```

## Results

30 test questions with answers checked against the 2026/27 rulebook text ([`eval.py`](eval.py)):
20 factual questions (including 2 off-topic ones that should be refused) and 10 match situations.
Retrieval and answers are scored separately:

- **Retrieval:** was the passage that answers the question in the top 5 chunks?
- **Answer:** does the final answer contain the right ruling?

| Chunking | Fact answers | Situation answers | Retrieval (all questions) |
|---|---|---|---|
| Fixed 200-word windows | 19/20 | 6/10 | 26/28 |
| **Section-aware, labelled chunks** | 17/20* | **7/10** | **27/28** |

The two off-topic questions were refused correctly in both versions (best reranker score ≈ −5.7, against +3 to +5 for real
rules questions).

**What I learned from the errors:**

- **Search was the first bottleneck.** My first prototype answered "Can a goalkeeper score from a goal kick?" using the
  *penalty kick* rules: "goalkeeper" plus "kick" matched the wrong Law. Adding a reranker and labelling each chunk with
  its Law fixed most of these. Section-aware chunks now find the right passage 27 times out of 28.
- **Now the language model is the bottleneck.** Most remaining mistakes happen *after* the right rule was retrieved.
  For example, it found the rule saying a ball that touches the referee and goes in is a **dropped ball**, but still
  answered "goal". A stronger LLM, or a prompt that makes it quote the rule first, is the next step.
- **Keyword grading is rough,** so every answer is also checked by hand. \*The automatic grader marked "Can a goal be
  scored directly from a goal kick?" as correct because its key word ("oppos") matched "awarded to the **oppo**nents";
  the answer actually said *no* (it's yes, against the opposing team, Law 16). The score above is the corrected one and
  the grader is now stricter. Lesson: check the answers, not just the score.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py          # chat app at http://127.0.0.1:7860 (downloads the rulebook and models on first run)
python eval.py         # runs the 30-question test for both chunking strategies
```

- `PREVIEW=1 python app.py` starts the app without the LLM (it shows the best-matching rule), which is handy on slow machines.
- `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct` uses a smaller model on CPU.
- A GPU (or an Apple Silicon Mac) is strongly recommended. The test above ran on a free Google Colab T4.

## Files

| File | What it does |
|---|---|
| `rag.py` | The pipeline: PDF loading, both chunking strategies, ChromaDB index, retrieval and reranking, answer generation |
| `eval.py` | The 30-question test, with a retrieval check and an answer check per question |
| `app.py` | Gradio chat app (custom dark design made in Figma) |
| `results/` | Test scores |

The rulebook isn't included in this repo. It's downloaded from [theifab.com](https://www.theifab.com/laws-of-the-game-documents/)
the first time you run it. The Laws of the Game are © The IFAB.

## Limitations

- It answers from the 2026/27 Laws only, not competition-specific rules or refereeing interpretations.
- Small open-source LLMs can misread a correctly retrieved rule, as in the results above, so always check the cited page.

---
Built by Abbas Ammar · [GitHub](https://github.com/abbasammar1)
