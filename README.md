# ⚽ Ref AI: a football rules assistant

Ask a question about the football **Laws of the Game 2026/27** and get an answer taken only from the official IFAB rulebook,
with the page it came from. If the rulebook doesn't cover it ("Who won the 2022 World Cup?"), it politely says it only answers
questions about the Laws of the Game, instead of guessing.



https://github.com/user-attachments/assets/d34e0806-446e-427a-a97c-6ca1c006c1e5


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

## Updates after code review (Nov 2026)

A mentor reviewed the project and pointed out three weak spots. Here's what changed:

1. **The index now rebuilds when the chunking changes.** Before, the ChromaDB index was only built when the collection
   was empty, so after editing the chunking code the *old* chunks in `.chroma/` were silently reused. Each collection
   name now includes a hash of the chunker's code, its settings and the embedding model (e.g. `laws_section_5a1c446c`),
   so any change builds a fresh index and outdated ones are deleted.
2. **Stricter answer grading, plus an LLM judge.** Loose keywords like "not" were removed (almost any negative sentence
   contains "not", so a wrong answer could pass). Questions now need specific phrases such as "no offside offence" or
   "cannot be scored", with forbidden phrases for the opposite ruling. `eval.py` also asks the LLM to compare each
   answer with the rulebook passage (LLM-as-judge) and lists every question where the judge and the keyword check
   disagree, for a manual look.
3. **Harder off-topic tests for the refusal threshold.** `MIN_SCORE` was chosen from only two off-topic questions, so I
   added five football questions the rulebook can't answer. Top reranker scores:

   | Question | Fixed | Section |
   |---|---|---|
   | Who refereed the 2022 World Cup final? | **+1.47** | **+1.34** |
   | Who is the best referee in the Premier League? | −3.54 | −4.69 |
   | What formation should my team play against a 4-3-3? | −7.40 | −6.74 |
   | How much money does a Premier League referee earn? | −8.69 | −9.78 |
   | Which club has won the most Champions League titles? | −10.63 | −10.72 |
   | *Lowest score of a real rules question* | *+0.34* | *+3.07* |

   **Finding:** the threshold alone is not enough. "Who refereed the 2022 World Cup final?" scores *above* `MIN_SCORE = 0`
   (it reads a lot like the referee sections of the rulebook), and with fixed chunks it even beats a real question
   (+0.34), so no single cut-off separates them. Such questions fall through to the second safety net: the system
   prompt tells the LLM to refuse when the excerpts don't contain the answer. `eval.py` now reports the highest
   off-topic score and the lowest answerable score on every run, so this gap is tracked instead of assumed.

The results table above is from the Nov 10 Colab run. Re-run `eval.py` to get scores with the stricter grader.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py          # chat app at http://127.0.0.1:7860 (downloads the rulebook and models on first run)
python eval.py         # runs the 35-question test for both chunking strategies
```

- `PREVIEW=1 python app.py` starts the app without the LLM (it shows the best-matching rule), which is handy on slow machines.
- `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct` uses a smaller model on CPU.
- A GPU (or an Apple Silicon Mac) is strongly recommended. The test above ran on a free Google Colab T4.

## Files

| File | What it does |
|---|---|
| `rag.py` | The pipeline: PDF loading, both chunking strategies, ChromaDB index, retrieval and reranking, answer generation |
| `eval.py` | The 35-question test: retrieval check, phrase-based answer check, LLM-as-judge, and a refusal-threshold check |
| `app.py` | Gradio chat app (custom dark design made in Figma) |
| `results/` | Test scores |

The rulebook isn't included in this repo. It's downloaded from [theifab.com](https://www.theifab.com/laws-of-the-game-documents/)
the first time you run it. The Laws of the Game are © The IFAB.

## Limitations

- It answers from the 2026/27 Laws only, not competition-specific rules or refereeing interpretations.
- Small open-source LLMs can misread a correctly retrieved rule, as in the results above, so always check the cited page.

---
Built by Abbas Ammar · [GitHub](https://github.com/abbasammar1)
