"""
Cache Generator for HallucinationGuard-Env
==========================================
Run this ONCE on your PC to download all datasets and save them as cache files.
Then upload the cache/ folder to your HF Space.

Usage:
    pip install datasets
    python generate_cache.py

Output:
    server/cache/squad_2000.json
    server/cache/trivia_qa_2000.json
    server/cache/halueval_1000.json
    ... etc (one file per dataset)

After this finishes, upload everything:
    python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='.', repo_id='SamSankar/hallucination-guard-env', repo_type='space', ignore_patterns=['__pycache__', '*.pyc']); print('Upload complete!')"
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    from datasets import load_dataset as hf_load
except ImportError:
    print("Run: pip install datasets")
    sys.exit(1)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

DATASETS = {
    "squad":         10000,
    "trivia_qa":     10000,
    "halueval":      5000,
    "truthful_qa":   817,
    "hotpotqa":      10000,
    "boolq":         9427,
    "faithdial":     10000,
    "fever":         10000,
    "arc":           3370,
    "openbookqa":    4957,
    "ms_marco":      10000,
    "coqa":          7199,
    "nq_open":       8000,
    "commonsense_qa": 8000,
    "winogrande":    5000,
}
# Target: ~100k examples

def load_squad(cap):
    ds = hf_load("squad", split=f"train[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        ans = item.get("answers", {}).get("text", [])
        answer = ans[0] if ans else ""
        if not answer or not item.get("context"): continue
        out.append({"question": item["question"], "context": item["context"][:1500],
                    "answer": answer, "id": f"squad_{i}", "source": "squad",
                    "difficulty": "intermediate", "category": "reading_comprehension",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out

def load_trivia_qa(cap):
    ds = hf_load("trivia_qa", "rc.wikipedia", split=f"train[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        cp = item.get("entity_pages", {})
        ctx = ""
        if isinstance(cp, dict):
            ctxs = cp.get("wiki_context", [])
            ctx = ctxs[0] if isinstance(ctxs, list) and ctxs else str(ctxs)
        if not ctx: continue
        aliases = item.get("answer", {}).get("normalized_aliases", [])
        answer = aliases[0] if aliases else item.get("answer", {}).get("value", "")
        if not answer: continue
        out.append({"question": item["question"], "context": ctx[:1500],
                    "answer": str(answer), "id": f"triviaqa_{i}", "source": "trivia_qa",
                    "difficulty": "intermediate", "category": "trivia",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out

def load_halueval(cap):
    ds = hf_load("pminervini/HaluEval", "qa", split=f"data[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        q = item.get("question", "")
        ctx = item.get("knowledge", item.get("context", ""))
        ans = item.get("right_answer", item.get("answer", ""))
        if not q or not ans: continue
        out.append({"question": q, "context": str(ctx)[:1500],
                    "answer": str(ans), "id": f"halueval_{i}", "source": "halueval",
                    "difficulty": "advanced", "category": "hallucination_detection",
                    "hallucination_type": item.get("hallucination_type"),
                    "entities": [], "metadata": {}})
    return out

def load_truthful_qa(cap):
    ds = hf_load("truthful_qa", "generation", split="validation")
    out = []
    for i, item in enumerate(ds):
        if i >= cap: break
        best = item.get("best_answer", "")
        correct = item.get("correct_answers", [])
        ctx = " ".join(correct) if correct else item.get("question", "")
        if not best: continue
        out.append({"question": item["question"], "context": ctx[:1500],
                    "answer": best, "id": f"truthfulqa_{i}", "source": "truthful_qa",
                    "difficulty": "expert", "category": "factuality",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out

def load_hotpotqa(cap):
    ds = hf_load("hotpot_qa", "fullwiki", split=f"train[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        q = item.get("question", "")
        ans = item.get("answer", "")
        titles = item.get("context", {}).get("title", [])
        sents  = item.get("context", {}).get("sentences", [])
        ctx = " ".join(f"{t}: {' '.join(s)}" for t, s in zip(titles, sents))[:1500]
        if not q or not ans or not ctx: continue
        out.append({"question": q, "context": ctx, "answer": str(ans),
                    "id": f"hotpotqa_{i}", "source": "hotpotqa",
                    "difficulty": "advanced", "category": "multi_hop_reasoning",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out

def load_boolq(cap):
    ds = hf_load("google/boolq", split=f"train[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        q = item.get("question", "")
        p = item.get("passage", "")
        if not q or not p: continue
        out.append({"question": q, "context": p[:1500],
                    "answer": "yes" if item.get("answer", False) else "no",
                    "id": f"boolq_{i}", "source": "boolq",
                    "difficulty": "beginner", "category": "yes_no_qa",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out

def load_faithdial(cap):
    # Anthropic HH-RLHF: human preference dialogues, 100% parquet
    ds = hf_load("Anthropic/hh-rlhf", split="train[:%d]" % cap)
    out = []
    for i, item in enumerate(ds):
        chosen = item.get("chosen", "")
        if not chosen:
            continue
        parts = chosen.split("Human:")
        question = ""
        answer = ""
        for part in parts[1:]:
            if "Assistant:" in part:
                q_part, a_part = part.split("Assistant:", 1)
                q = q_part.strip()
                a = a_part.split("Human:")[0].strip()
                if q and a:
                    question = q
                    answer = a
        if not question or not answer:
            continue
        ctx = chosen[:800]
        out.append({
            "question": question[:200],
            "context": ctx,
            "answer": answer[:400],
            "id": "faithdial_%d" % i,
            "source": "faithdial",
            "difficulty": "advanced",
            "category": "hallucination_detection",
            "hallucination_type": None,
            "entities": [],
            "metadata": {}
        })
    return out

def load_fever(cap):
    # Stanford NLI: entailment/contradiction/neutral — pure parquet, fact verification task
    ds = hf_load("stanfordnlp/snli", split=f"train[:{cap}]")
    label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO", -1: "NOT ENOUGH INFO"}
    out = []
    for i, item in enumerate(ds):
        premise    = item.get("premise", "")
        hypothesis = item.get("hypothesis", "")
        label_int  = item.get("label", -1)
        label      = label_map.get(int(label_int), "NOT ENOUGH INFO")
        if not premise or not hypothesis or label_int == -1: continue
        out.append({"question": f"Does the premise support or refute this hypothesis? Hypothesis: {hypothesis}",
                    "context": f"Premise: {premise}", "answer": label,
                    "id": f"fever_{i}", "source": "fever",
                    "difficulty": "advanced", "category": "fact_verification",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out


def load_arc(cap):
    # Combine train + validation + test to get full ~3370 examples
    out = []
    for split in ["train", "validation", "test"]:
        try:
            ds = hf_load("allenai/ai2_arc", "ARC-Challenge", split=split)
            for item in ds:
                if len(out) >= cap: break
                q = item.get("question", "")
                choices = item.get("choices", {})
                ans_key = item.get("answerKey", "")
                labels = choices.get("label", [])
                texts  = choices.get("text", [])
                ctx = "Choices: " + " | ".join(f"{l}: {t}" for l, t in zip(labels, texts))
                answer = next((t for l, t in zip(labels, texts) if l == ans_key), "")
                if not q or not answer: continue
                out.append({"question": q, "context": ctx, "answer": answer,
                            "id": f"arc_{len(out)}", "source": "arc",
                            "difficulty": "advanced", "category": "science_exam",
                            "hallucination_type": None, "entities": [], "metadata": {}})
        except Exception:
            continue
    return out

def load_openbookqa(cap):
    ds = hf_load("allenai/openbookqa", "main", split=f"train[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        q = item.get("question_stem", "")
        choices = item.get("choices", {})
        ans_key = item.get("answerKey", "")
        labels = choices.get("label", [])
        texts  = choices.get("text", [])
        fact   = item.get("fact1", "")
        ctx = f"Core fact: {fact} | Choices: " + " | ".join(f"{l}: {t}" for l, t in zip(labels, texts))
        answer = next((t for l, t in zip(labels, texts) if l == ans_key), "")
        if not q or not answer: continue
        out.append({"question": q, "context": ctx[:1500], "answer": answer,
                    "id": f"openbookqa_{i}", "source": "openbookqa",
                    "difficulty": "intermediate", "category": "science_facts",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out

def load_ms_marco(cap):
    ds = hf_load("microsoft/ms_marco", "v2.1", split=f"train[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        q = item.get("query", "")
        passages = item.get("passages", {})
        texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
        ctx = " ".join(texts)[:1500] if texts else ""
        answers = item.get("answers", [])
        answer = answers[0] if answers else ""
        if not q or not ctx or not answer or answer == "No Answer Present.": continue
        out.append({"question": q, "context": ctx, "answer": str(answer),
                    "id": f"msmarco_{i}", "source": "ms_marco",
                    "difficulty": "intermediate", "category": "web_search_qa",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out

def load_coqa(cap):
    ds = hf_load("stanfordnlp/coqa", split=f"train[:{cap}]")
    out = []
    for i, item in enumerate(ds):
        story = item.get("story", "")
        questions = item.get("questions", [])
        answers = item.get("answers", {})
        ans_texts = answers.get("input_text", []) if isinstance(answers, dict) else []
        if not story or not questions or not ans_texts: continue
        q = questions[0] if questions else ""
        answer = ans_texts[0] if ans_texts else ""
        if not q or not answer: continue
        out.append({"question": str(q), "context": story[:1500], "answer": str(answer),
                    "id": f"coqa_{i}", "source": "coqa",
                    "difficulty": "intermediate", "category": "conversational_qa",
                    "hallucination_type": None, "entities": [], "metadata": {}})
    return out


def load_nq_open(cap):
    ds = hf_load("nq_open", split="train[:%d]" % cap)
    out = []
    for i, item in enumerate(ds):
        q = item.get("question", "")
        answers = item.get("answer", [])
        answer = answers[0] if answers else ""
        if not q or not answer:
            continue
        out.append({
            "question": q,
            "context": "Answer based on your knowledge: " + q,
            "answer": str(answer),
            "id": "nq_open_%d" % i,
            "source": "nq_open",
            "difficulty": "intermediate",
            "category": "open_domain_qa",
            "hallucination_type": None,
            "entities": [],
            "metadata": {}
        })
    return out


def load_commonsense_qa(cap):
    ds = hf_load("tau/commonsense_qa", split="train[:%d]" % cap)
    out = []
    for i, item in enumerate(ds):
        q = item.get("question", "")
        choices = item.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        texts  = choices.get("text", []) if isinstance(choices, dict) else []
        ans_key = item.get("answerKey", "")
        ctx = "Choices: " + " | ".join(
            "%s: %s" % (l, t) for l, t in zip(labels, texts))
        answer = next((t for l, t in zip(labels, texts) if l == ans_key), "")
        if not q or not answer:
            continue
        out.append({
            "question": q,
            "context": ctx,
            "answer": answer,
            "id": "csqa_%d" % i,
            "source": "commonsense_qa",
            "difficulty": "intermediate",
            "category": "commonsense_reasoning",
            "hallucination_type": None,
            "entities": [],
            "metadata": {}
        })
    return out


def load_winogrande(cap):
    # WinoGrande — commonsense reasoning, 100% parquet, no scripts
    ds = hf_load("allenai/winogrande", "winogrande_xl", split="train[:%d]" % cap)
    out = []
    for i, item in enumerate(ds):
        sentence = item.get("sentence", "")
        opt1 = item.get("option1", "")
        opt2 = item.get("option2", "")
        answer_key = str(item.get("answer", "1"))
        answer = opt1 if answer_key == "1" else opt2
        if not sentence or not answer:
            continue
        ctx = "Sentence: %s Options: 1: %s | 2: %s" % (sentence, opt1, opt2)
        out.append({
            "question": "Which option correctly fills the blank? " + sentence,
            "context": ctx,
            "answer": answer,
            "id": "winogrande_%d" % i,
            "source": "winogrande",
            "difficulty": "intermediate",
            "category": "commonsense_reasoning",
            "hallucination_type": None,
            "entities": [],
            "metadata": {}
        })
    return out



LOADERS = {
    "squad":        load_squad,
    "trivia_qa":    load_trivia_qa,
    "halueval":     load_halueval,
    "truthful_qa":  load_truthful_qa,
    "hotpotqa":     load_hotpotqa,
    "boolq":        load_boolq,
    "faithdial":    load_faithdial,
    "fever":        load_fever,
    "arc":          load_arc,
    "openbookqa":   load_openbookqa,
    "ms_marco":     load_ms_marco,
    "coqa":         load_coqa,
    "nq_open":        load_nq_open,
    "commonsense_qa": load_commonsense_qa,
    "winogrande":     load_winogrande,
}

def main():
    total = 0
    print(f"\n{'='*55}")
    print(f"  HallucinationGuard Cache Generator")
    print(f"  Output: {CACHE_DIR}")
    print(f"{'='*55}\n")

    for ds_name, cap in DATASETS.items():
        cache_file = os.path.join(CACHE_DIR, f"{ds_name}_{cap}.json")

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                existing = json.load(f)
            print(f"  ✅ {ds_name}: already cached ({len(existing)} examples) — skipping")
            total += len(existing)
            continue

        print(f"  Downloading {ds_name} ({cap} examples)...", end=" ", flush=True)
        t0 = time.time()
        try:
            loader = LOADERS[ds_name]
            examples = loader(cap)
            with open(cache_file, "w") as f:
                json.dump(examples, f)
            elapsed = time.time() - t0
            total += len(examples)
            print(f"✅ {len(examples)} examples saved ({elapsed:.0f}s)")
        except Exception as e:
            print(f"❌ Failed: {e}")

    print(f"\n{'='*55}")
    print(f"  Done! Total examples cached: {total:,}")
    print(f"  Cache location: {CACHE_DIR}")
    print(f"\n  Now upload to HF Space:")
    print(f"  python -c \"from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='.', repo_id='SamSankar/hallucination-guard-env', repo_type='space', ignore_patterns=['__pycache__', '*.pyc']); print('Upload complete!')\"")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
