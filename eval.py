"""Evaluate the assistant on 30 questions with known answers from the 2026/27 rulebook.

For every question we measure two things separately:
  - retrieval: did one of the top 5 chunks contain the sentence that answers it?
  - answer:    does the final answer contain the right key words (and none of the wrong ones)?

Keyword grading is rough, so results/answers_<strategy>.md lists every answer for a human check.

Usage:  python eval.py            # evaluates both chunking strategies
        python eval.py section    # just one
"""

import json
import sys
import time
from pathlib import Path

from rag import RulesAssistant, load_pages, normalize

# (question, needle: text in the rulebook that answers it, keywords: any must appear, forbidden: none may appear)
FACTS = [
    ("How many players can each team have on the field at the start of a match?", "maximum of eleven players", ["11", "eleven"], []),
    ("What is the minimum number of players a team needs for a match to start or continue?", "fewer than seven players", ["7", "seven"], []),
    ("How long is each half of a match?", "two equal halves of 45 minutes", ["45"], []),
    ("How long can the half-time interval be?", "not exceeding 15 minutes", ["15"], []),
    ("What is the circumference of the ball?", "68 cm", ["68", "70"], []),
    ("How far is the penalty mark from the goal line?", "11 m", ["11"], []),
    ("How wide is the goal?", "7.32", ["7.32"], []),
    ("How high is the crossbar?", "2.44", ["2.44"], []),
    ("Can a goal be scored directly from a throw-in?", "a goal cannot be scored directly from a throw-in", ["cannot", "not"], []),
    ("Can a goal be scored directly from a goal kick?", "a goal may be scored directly from a goal kick", ["yes", "may be scored", "can score", "can be scored"], ["cannot", "can't", "not be scored"]),
    ("Is there an offside offence if a player receives the ball directly from a throw-in?", "no offside offence if a player receives the ball directly from", ["not", "no offside", "no offence"], []),
    ("Is there an offside offence if a player receives the ball directly from a corner kick?", "no offside offence if a player receives the ball directly from", ["not", "no offside", "no offence"], []),
    ("How far must opponents be from the ball at a free kick?", "9.15 m", ["9.15"], []),
    ("What colour card does the referee show for a caution?", "the yellow card communicates a caution", ["yellow"], []),
    ("What colour card is shown when a player is sent off?", "the red card communicates a sending-off", ["red"], []),
    ("How does the referee signal an indirect free kick?", "raising the arm above the head", ["arm"], []),
    ("Where must the goalkeeper be when a penalty kick is taken?", "goal line", ["goal line"], []),
    ("How long can a goalkeeper hold the ball in their hands?", "more than eight seconds", ["8", "eight"], []),
    ("Who won the 2022 World Cup?", None, None, []),
    ("What is the capital of France?", None, None, []),
]

SITUATIONS = [
    ("A defender deliberately handles the ball on the goal line and stops a certain goal. What is the sanction?",
     "by committing a deliberate handball offence", ["sent off", "sending-off", "sending off", "red card"], []),
    ("A defender fouls an attacker in the penalty area while trying to play the ball, denying an obvious goal-scoring opportunity. What card is shown?",
     "attempt to play the ball or a challenge for the ball and the referee awards a penalty kick", ["caution", "yellow"], ["is sent off", "be sent off", "red card is shown"]),
    ("Is a player penalised for offside if they receive the ball directly from a goal kick?",
     "no offside offence if a player receives the ball directly from", ["not", "no offside", "no offence"], []),
    ("What happens if a player scores directly from an indirect free kick into the opponents' goal?",
     "if an indirect free kick is kicked directly into the opponents' goal, a goal kick is awarded", ["goal kick"], []),
    ("The ball hits the referee and goes into the goal. What happens?",
     "touches a match official, remains on the field of play", ["dropped ball", "drop"], ["goal is awarded", "goal stands", "goal counts"]),
    ("What happens if the goalkeeper holds the ball for more than eight seconds?",
     "more than eight seconds", ["corner"], []),
    ("At a penalty kick, the kicker touches the ball a second time before any other player touches it. What is awarded?",
     "the kicker touches the ball again before it has touched another player", ["indirect"], []),
    ("A throw-in is thrown directly into the thrower's own goal. What is awarded?",
     "if the ball enters the thrower's goal - a corner kick is awarded", ["corner"], []),
    ("A goal kick goes directly into the kicker's own goal. What is awarded?",
     "if the ball directly enters the kicker's goal, a corner kick is awarded", ["corner"], []),
    ("What happens if a player receives a second caution in the same match?",
     "receiving a second caution in the same match", ["sent off", "sending-off", "sending off", "red"], []),
]

TESTS = [("fact", *t) for t in FACTS] + [("situation", *t) for t in SITUATIONS]
RESULTS = Path(__file__).parent / "results"


def check_needles():
    """Make sure every needle really is in the rulebook, so the retrieval check is meaningful."""
    full = normalize(" ".join(p["text"] for p in load_pages())).lower().replace("–", "-")
    missing = [q for _, q, needle, *_ in TESTS if needle and normalize(needle).lower() not in full]
    for q in missing:
        print("WARNING: needle not found in rulebook for:", q)
    return missing


def clean(text):
    return normalize(text).lower().replace("–", "-")


def grade_answer(answer, keywords, forbidden):
    a = clean(answer)
    if keywords is None:
        return "can't help" in a or "don't know" in a or "do not know" in a
    return any(k in a for k in keywords) and not any(f in a for f in forbidden)


def evaluate(strategy):
    assistant = RulesAssistant(strategy=strategy)
    rows = []
    for kind, question, needle, keywords, forbidden in TESTS:
        start = time.time()
        result = assistant.ask(question)
        retrieved = " ".join(clean(h["text"]) for h in result["sources"])
        rows.append({
            "kind": kind,
            "question": question,
            "answer": result["answer"],
            "pages": [h["page"] for h in result["sources"]],
            "top_score": round(result["sources"][0]["score"], 2) if result["sources"] else None,
            "retrieval_hit": None if needle is None else clean(needle) in retrieved,
            "answer_correct": grade_answer(result["answer"], keywords, forbidden),
            "seconds": round(time.time() - start, 1),
        })
        mark = "OK " if rows[-1]["answer_correct"] else "BAD"
        print(f"[{strategy}] {mark} {question[:70]}")
    return rows


def summarize(rows):
    out = {}
    for kind in ("fact", "situation"):
        part = [r for r in rows if r["kind"] == kind]
        with_needle = [r for r in part if r["retrieval_hit"] is not None]
        out[kind] = {
            "answers": f"{sum(r['answer_correct'] for r in part)}/{len(part)}",
            "retrieval": f"{sum(r['retrieval_hit'] for r in with_needle)}/{len(with_needle)}",
        }
    return out


def write_report(strategy, rows):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_{strategy}.json").write_text(json.dumps(rows, indent=2))
    lines = [f"# Answers: `{strategy}` chunking\n"]
    for r in rows:
        mark = "✅" if r["answer_correct"] else "❌"
        hit = {True: "found", False: "missed", None: "n/a"}[r["retrieval_hit"]]
        lines.append(f"### {mark} {r['question']}\n")
        lines.append(f"{r['answer']}\n")
        lines.append(f"*Pages: {r['pages']} · top score {r['top_score']} · answer passage {hit}*\n")
    (RESULTS / f"answers_{strategy}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    check_needles()
    strategies = sys.argv[1:] or ["fixed", "section"]
    summary = {}
    for s in strategies:
        rows = evaluate(s)
        write_report(s, rows)
        summary[s] = summarize(rows)
    print(json.dumps(summary, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
