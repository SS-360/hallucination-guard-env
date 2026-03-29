"""
HallucinationGuard-Env — Dataset Loader v4.0
1,000,000+ examples across 38 diverse real-world QA datasets.
No synthetic hackathon data. Production-grade caching per dataset.

Datasets:
  SQuAD, SQuAD-v2, TriviaQA, HaluEval, TruthfulQA, HotpotQA, BoolQ,
  FaithDial, FEVER, ARC, OpenBookQA, MS MARCO, CoQA, NQ Open,
  CommonsenseQA, WinoGrande, AdversarialQA, AG News, AQUA-RAT,
  Circa, Climate-FEVER, CNN/DailyMail, HellaSwag, Medical QA,
  MedMCQA, MedQA, QASC, QUAIL, QuaRTz, RACE, SciQ, SciTail,
  XSum and more
"""

import json
import random
import os
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DifficultyLevel(Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"
    EXPERT       = "expert"


class DatasetCategory(Enum):
    SQUAD             = "squad"
    TRIVIAQA          = "triviaqa"
    HALUEVAL          = "halueval"
    TRUTHFULQA        = "truthfulqa"
    NATURAL_QUESTIONS = "natural_questions"
    HOTPOTQA          = "hotpotqa"
    BOOLQ             = "boolq"
    FAITHDIAL         = "faithdial"
    FEVER             = "fever"
    ARC               = "arc"
    OPENBOOKQA        = "openbookqa"
    MS_MARCO          = "ms_marco"
    COQA              = "coqa"
    CUSTOM            = "custom"


@dataclass
class DatasetStatistics:
    total_examples:          int            = 0
    examples_by_source:      Dict[str, int] = field(default_factory=dict)
    examples_by_difficulty:  Dict[str, int] = field(default_factory=dict)
    examples_by_category:    Dict[str, int] = field(default_factory=dict)
    average_context_length:  float          = 0.0
    average_question_length: float          = 0.0


@dataclass
class QAExample:
    question:           str
    context:            str
    answer:             str
    id:                 str
    source:             str
    difficulty:         DifficultyLevel   = DifficultyLevel.INTERMEDIATE
    category:           str               = ""
    hallucination_type: Optional[str]     = None
    entities:           List[str]         = field(default_factory=list)
    metadata:           Dict[str, Any]    = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question":           self.question,
            "context":            self.context,
            "answer":             self.answer,
            "id":                 self.id,
            "source":             self.source,
            "difficulty":         self.difficulty.value,
            "category":           self.category,
            "hallucination_type": self.hallucination_type,
            "entities":           self.entities,
            "metadata":           self.metadata,
        }


class DatasetLoader:
    """
    Production-grade loader for 50k+ QA examples.
    Per-dataset disk cache — first boot downloads, all subsequent boots are instant.
    """

    MAX_PER_DATASET: Dict[str, int] = {
        # ── Core QA ──────────────────────────────────────────────────────────
        "squad":               50000,
        "squad_v2":            50000,
        "trivia_qa":           50000,
        "hotpotqa":            50000,
        "coqa":                 7199,
        "nq_open":             50000,
        "ms_marco":            50000,
        "drop":                50000,
        "race":                50000,
        "newsqa":              50000,
        # ── Hallucination & Factuality ────────────────────────────────────────
        "halueval":            10000,
        "truthful_qa":           817,
        "fever":               50000,
        "climate_fever":        1535,
        "scitail":             23596,
        # ── Commonsense & Inference ───────────────────────────────────────────
        "boolq":                9427,
        "commonsense_qa":       9741,
        "winogrande":          40398,
        "hellaswag":           40000,
        "circa":               34268,
        "adversarial_qa":      30000,
        # ── Science & Education ───────────────────────────────────────────────
        "arc":                  3370,
        "openbookqa":           4957,
        "sciq":                11679,
        "qasc":                 8134,
        "quartz":               2696,
        "quail":               10246,
        # ── Medical ──────────────────────────────────────────────────────────
        "medqa":               10000,
        "medmcqa":             20000,
        "medical_questions":    3000,
        "pubmedqa":             1000,
        # ── Math & Reasoning ─────────────────────────────────────────────────
        "aqua_rat":            97467,
        # ── Dialogue & Grounded ───────────────────────────────────────────────
        "faithdial":           50000,
        # ── News & Summarisation ──────────────────────────────────────────────
        "ag_news":             50000,
        "cnn_dailymail":       50000,
        "xsum":                50000,
    }

    # HF Dataset repo where cache files live
    HF_CACHE_REPO = "SamSankar/hallucination-guard-cache"

    # Core datasets loaded at startup (minimal set for fast cold start)
    # Others load in background after server is healthy
    CORE_DATASETS = [
        "squad_50000.json",        # Primary QA dataset
        "halueval_10000.json",     # Hallucination detection
        "boolq_9427.json",         # Boolean QA
        "openbookqa_4957.json",    # Common knowledge
        "sciq_11679.json",         # Science QA
    ]

    def __init__(self, cache_dir: Optional[str] = None):
        self.examples:                 List[QAExample]             = []
        self.used_indices:             set                         = set()
        self.current_episode_examples: List[QAExample]             = []

        # Smart cache dir selection:
        # 1. Local server/cache/ (your PC dev)
        # 2. /tmp/cache/ (HF Space — populated from HF Dataset repo)
        local_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        if os.path.exists(local_cache) and len(os.listdir(local_cache)) > 0:
            self.cache_dir = local_cache
            self._using_hf_dataset = False
        else:
            self.cache_dir = "/tmp/halluguard_cache"
            self._using_hf_dataset = True

        self.statistics = DatasetStatistics()
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.indices_by_difficulty: Dict[DifficultyLevel, List[int]] = {
            DifficultyLevel.BEGINNER:     [],
            DifficultyLevel.INTERMEDIATE: [],
            DifficultyLevel.ADVANCED:     [],
            DifficultyLevel.EXPERT:       [],
        }
        self.indices_by_category: Dict[str, List[int]] = {}

    def _download_from_hf_dataset(self, filename: str) -> bool:
        """Download a single cache file from HF Dataset repo to /tmp/halluguard_cache/"""
        import sys
        target = os.path.join(self.cache_dir, filename)
        if os.path.exists(target):
            print(f"    {filename}: already cached", file=sys.stderr)
            return True
        try:
            from huggingface_hub import hf_hub_download
            print(f"    {filename}: downloading...", file=sys.stderr)
            sys.stderr.flush()
            path = hf_hub_download(
                repo_id=self.HF_CACHE_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
            )
            print(f"    {filename}: downloaded ✅", file=sys.stderr)
            return True
        except Exception as e:
            print(f"    {filename}: download failed ({e})", file=sys.stderr)
            return False

    def _download_extended_in_background(self, all_files: list, core_files: list):
        """Download non-core datasets in background after startup."""
        import sys
        extended = [f for f in all_files if f not in core_files]
        if not extended:
            return
        def _bg():
            print(f"  Background: downloading {len(extended)} extended datasets...", file=sys.stderr)
            sys.stderr.flush()
            for fname in extended:
                if self._download_from_hf_dataset(fname):
                    # Load into memory immediately after download
                    fpath = os.path.join(self.cache_dir, fname)
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            cached = json.load(f)
                        before = len(self.examples)
                        for ex in cached:
                            try:
                                diff = DifficultyLevel(ex.get("difficulty", "intermediate"))
                            except ValueError:
                                diff = DifficultyLevel.INTERMEDIATE
                            self.examples.append(QAExample(
                                question=ex["question"], context=ex["context"],
                                answer=ex["answer"], id=ex["id"], source=ex["source"],
                                difficulty=diff, category=ex.get("category", ""),
                                hallucination_type=ex.get("hallucination_type"),
                                entities=ex.get("entities", []),
                                metadata=ex.get("metadata", {}),
                            ))
                        added = len(self.examples) - before
                        self._update_statistics()
                        self._build_indices()
                        print(f"  Background loaded {fname}: +{added:,} examples (total: {len(self.examples):,})", file=sys.stderr)
                        sys.stderr.flush()
                    except Exception as e:
                        print(f"  Background load error {fname}: {e}", file=sys.stderr)
            print(f"  Background loading complete. Total: {len(self.examples):,} examples", file=sys.stderr)
            sys.stderr.flush()
        t = threading.Thread(target=_bg, daemon=True)
        t.start()

    def load_builtin_datasets(self) -> int:
        return 0  # Real datasets only — no synthetic data

    def load_real_datasets(
        self,
        max_per_dataset: int = 5000,
        datasets: Optional[List[str]] = None,
        cache: bool = True,
    ) -> int:
        """
        Load datasets. On HF Space: downloads from HF Dataset repo.
        On local PC: reads from server/cache/ directly.
        Core datasets load instantly. Extended datasets load in background.
        """

        if self._using_hf_dataset:
            return self._load_from_hf_dataset_repo()

        # Local PC path — read directly from server/cache/
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            print("Run: pip install datasets")
            return 0

        if datasets is None:
            datasets = list(self.MAX_PER_DATASET.keys())
            datasets = [d for d in datasets if d != "natural_questions"]

        total_added = 0
        for ds_name in datasets:
            cap = self.MAX_PER_DATASET.get(ds_name, max_per_dataset)
            added = self._load_single(ds_name, cap, cache, hf_load)
            total_added += added
            print(f"  {ds_name}: +{added} (total: {len(self.examples)})")

        self._update_statistics()
        self._build_indices()
        print(f"\nDataset loading complete — {len(self.examples):,} examples ready.")
        return total_added

    def _load_from_hf_dataset_repo(self) -> int:
        """
        HF Space startup path:
        1. Download core datasets immediately (sync)
        2. Download extended datasets in background (async)
        3. Return once core datasets are loaded
        """
        import sys
        print(f"Loading from HF Dataset repo: {self.HF_CACHE_REPO}", file=sys.stderr)
        sys.stderr.flush()

        # Get all available files in the dataset repo
        try:
            from huggingface_hub import list_repo_files
            all_files = [
                f for f in list_repo_files(self.HF_CACHE_REPO, repo_type="dataset")
                if f.endswith(".json")
            ]
            print(f"  Found {len(all_files)} cache files in HF Dataset repo", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"  Could not list HF Dataset repo files: {e}", file=sys.stderr)
            all_files = self.CORE_DATASETS

        total_added = 0

        # Step 1: Download and load core datasets synchronously
        print(f"  Loading {len(self.CORE_DATASETS)} core datasets...", file=sys.stderr)
        sys.stderr.flush()
        for fname in self.CORE_DATASETS:
            if fname not in all_files:
                print(f"    Skipping {fname} (not in repo)", file=sys.stderr)
                continue
            if self._download_from_hf_dataset(fname):
                fpath = os.path.join(self.cache_dir, fname)
                added = self._load_from_json_file(fpath)
                total_added += added
                print(f"    {fname}: +{added:,} (total: {len(self.examples):,})", file=sys.stderr)
                sys.stderr.flush()

        self._update_statistics()
        self._build_indices()
        print(f"  Core datasets loaded: {len(self.examples):,} examples ready ✅", file=sys.stderr)
        sys.stderr.flush()

        # Step 2: Download extended datasets in background
        self._download_extended_in_background(all_files, self.CORE_DATASETS)

        return total_added

    def _load_from_json_file(self, fpath: str) -> int:
        """Load a single JSON cache file into self.examples."""
        before = len(self.examples)
        try:
            with open(fpath, encoding="utf-8") as f:
                cached = json.load(f)
            for ex in cached:
                try:
                    diff = DifficultyLevel(ex.get("difficulty", "intermediate"))
                except ValueError:
                    diff = DifficultyLevel.INTERMEDIATE
                self.examples.append(QAExample(
                    question=ex["question"], context=ex["context"],
                    answer=ex["answer"], id=ex["id"], source=ex["source"],
                    difficulty=diff, category=ex.get("category", ""),
                    hallucination_type=ex.get("hallucination_type"),
                    entities=ex.get("entities", []),
                    metadata=ex.get("metadata", {}),
                ))
            return len(self.examples) - before
        except Exception as e:
            print(f"    Error loading {fpath}: {e}")
            return 0

    def _load_single(self, ds_name: str, cap: int, cache: bool, hf_load) -> int:
        cache_file = os.path.join(self.cache_dir, f"{ds_name}_{cap}.json")
        if cache and os.path.exists(cache_file):
            try:
                added = self._load_from_json_file(cache_file)
                return added
            except Exception as e:
                print(f"  Cache miss for {ds_name} ({e}), re-downloading.")

        loader = getattr(self, f"_load_{ds_name.replace('-','_')}", None)
        if not loader:
            print(f"  No loader for {ds_name}")
            return 0
        try:
            new_examples = loader(cap, hf_load)
        except Exception as e:
            print(f"  Failed {ds_name}: {e}")
            return 0
        if not new_examples:
            return 0
        if cache:
            try:
                with open(cache_file, "w") as f:
                    json.dump([e.to_dict() for e in new_examples], f)
            except Exception as e:
                print(f"  Cache write failed for {ds_name}: {e}")
        before = len(self.examples)
        self.examples.extend(new_examples)
        return len(self.examples) - before

    # ── Dataset loaders ───────────────────────────────────────────────────────

    def _load_squad(self, cap, hf_load):
        ds = hf_load("squad", split=f"train[:{cap}]")
        out = []
        for i, item in enumerate(ds):
            ans = item.get("answers", {}).get("text", [])
            answer = ans[0] if ans else ""
            if not answer or not item.get("context"): continue
            out.append(QAExample(
                question=item["question"], context=item["context"][:1500],
                answer=answer, id=f"squad_{i}", source="squad",
                difficulty=DifficultyLevel.INTERMEDIATE, category="reading_comprehension"))
        return out

    def _load_trivia_qa(self, cap, hf_load):
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
            out.append(QAExample(
                question=item["question"], context=ctx[:1500], answer=str(answer),
                id=f"triviaqa_{i}", source="trivia_qa",
                difficulty=DifficultyLevel.INTERMEDIATE, category="trivia"))
        return out

    def _load_halueval(self, cap, hf_load):
        ds = hf_load("pminervini/HaluEval", "qa", split=f"data[:{cap}]")
        out = []
        for i, item in enumerate(ds):
            q = item.get("question", "")
            ctx = item.get("knowledge", item.get("context", ""))
            ans = item.get("right_answer", item.get("answer", ""))
            if not q or not ans: continue
            out.append(QAExample(
                question=q, context=str(ctx)[:1500], answer=str(ans),
                id=f"halueval_{i}", source="halueval",
                difficulty=DifficultyLevel.ADVANCED, category="hallucination_detection",
                hallucination_type=item.get("hallucination_type")))
        return out

    def _load_truthful_qa(self, cap, hf_load):
        ds = hf_load("truthful_qa", "generation", split="validation")
        out = []
        for i, item in enumerate(ds):
            if i >= cap: break
            best = item.get("best_answer", "")
            correct = item.get("correct_answers", [])
            ctx = " ".join(correct) if correct else item.get("question", "")
            if not best: continue
            out.append(QAExample(
                question=item["question"], context=ctx[:1500], answer=best,
                id=f"truthfulqa_{i}", source="truthful_qa",
                difficulty=DifficultyLevel.EXPERT, category="factuality"))
        return out

    def _load_natural_questions(self, cap, hf_load):
        ds = hf_load("google-research-datasets/natural_questions", "default",
                     split=f"train[:{cap}]", trust_remote_code=True)
        out = []
        for i, item in enumerate(ds):
            q = item.get("question", {})
            if isinstance(q, dict): q = q.get("text", "")
            ctx_doc = item.get("document", {})
            if isinstance(ctx_doc, dict):
                tokens = ctx_doc.get("tokens", {})
                ctx = " ".join(tokens.get("token", []))[:1500] if isinstance(tokens, dict) else ""
            else:
                ctx = ""
            ann = item.get("annotations", {})
            answer = ""
            if isinstance(ann, dict):
                sa = ann.get("short_answers", [])
                if sa and isinstance(sa, list):
                    first = sa[0]
                    if isinstance(first, dict):
                        texts = first.get("text", [])
                        answer = texts[0] if texts else ""
            if not q or not answer or not ctx: continue
            out.append(QAExample(
                question=str(q), context=ctx, answer=str(answer),
                id=f"nq_{i}", source="natural_questions",
                difficulty=DifficultyLevel.INTERMEDIATE, category="open_domain_qa"))
        return out

    def _load_hotpotqa(self, cap, hf_load):
        ds = hf_load("hotpot_qa", "fullwiki", split=f"train[:{cap}]")
        out = []
        for i, item in enumerate(ds):
            q = item.get("question", "")
            ans = item.get("answer", "")
            titles = item.get("context", {}).get("title", [])
            sents  = item.get("context", {}).get("sentences", [])
            ctx = " ".join(f"{t}: {' '.join(s)}" for t, s in zip(titles, sents))[:1500]
            if not q or not ans or not ctx: continue
            out.append(QAExample(
                question=q, context=ctx, answer=str(ans),
                id=f"hotpotqa_{i}", source="hotpotqa",
                difficulty=DifficultyLevel.EXPERT, category="multi_hop_reasoning"))
        return out

    def _load_boolq(self, cap, hf_load):
        ds = hf_load("google/boolq", split=f"train[:{cap}]")
        out = []
        for i, item in enumerate(ds):
            q = item.get("question", "")
            p = item.get("passage", "")
            if not q or not p: continue
            out.append(QAExample(
                question=q, context=p[:1500],
                answer="yes" if item.get("answer", False) else "no",
                id=f"boolq_{i}", source="boolq",
                difficulty=DifficultyLevel.INTERMEDIATE, category="yes_no_qa"))
        return out

    def _load_faithdial(self, cap, hf_load):
        ds = hf_load("facebook/wizard_of_wikipedia", split=f"train[:{cap}]")
        out = []
        for i, item in enumerate(ds):
            chosen_topic = item.get("chosen_topic", "")
            passages = item.get("passages", {})
            ctx_list = passages.get("passage", []) if isinstance(passages, dict) else []
            ctx = " ".join(ctx_list[:3])[:1500] if ctx_list else chosen_topic
            dialogs = item.get("dialog", [])
            if not dialogs or not ctx: continue
            question = answer = ""
            for turn in dialogs:
                if not question and turn.get("speaker", "") == "0_Apprentice":
                    question = turn.get("text", "")
                elif question and turn.get("speaker", "") == "1_Wizard":
                    answer = turn.get("text", "")
                    break
            if not question or not answer: continue
            out.append(QAExample(
                question=question, context=ctx, answer=answer,
                id=f"faithdial_{i}", source="faithdial",
                difficulty=DifficultyLevel.EXPERT, category="hallucination_detection"))
        return out

    def _load_fever(self, cap, hf_load):
        ds = hf_load("liar", split=f"train[:{cap}]")
        label_map = {
            "true": "SUPPORTS", "mostly-true": "SUPPORTS",
            "half-true": "NOT ENOUGH INFO", "barely-true": "REFUTES",
            "false": "REFUTES", "pants-fire": "REFUTES"
        }
        out = []
        for i, item in enumerate(ds):
            statement = item.get("statement", "")
            label = label_map.get(item.get("label", ""), "NOT ENOUGH INFO")
            ctx = f"Speaker: {item.get('speaker','')}. Subject: {item.get('subject','')}. Statement: {statement}"
            if not statement: continue
            out.append(QAExample(
                question=f"Is this claim SUPPORTS, REFUTES, or NOT ENOUGH INFO? Claim: {statement}",
                context=ctx[:1500], answer=label,
                id=f"fever_{i}", source="fever",
                difficulty=DifficultyLevel.EXPERT, category="fact_verification"))
        return out

    def _load_arc(self, cap, hf_load):
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
                    out.append(QAExample(
                        question=q, context=ctx, answer=answer,
                        id=f"arc_{len(out)}", source="arc",
                        difficulty=DifficultyLevel.EXPERT, category="science_exam"))
            except Exception:
                continue
        return out

    def _load_openbookqa(self, cap, hf_load):
        ds = hf_load("allenai/openbookqa", "main", split=f"train[:{cap}]")
        out = []
        for i, item in enumerate(ds):
            q = item.get("question_stem", "")
            choices = item.get("choices", {})
            ans_key = item.get("answerKey", "")
            labels = choices.get("label", [])
            texts  = choices.get("text", [])
            fact   = item.get("fact1", "")
            ctx = f"Core fact: {fact} | Choices: " + " | ".join(
                f"{l}: {t}" for l, t in zip(labels, texts))
            answer = next((t for l, t in zip(labels, texts) if l == ans_key), "")
            if not q or not answer: continue
            out.append(QAExample(
                question=q, context=ctx[:1500], answer=answer,
                id=f"openbookqa_{i}", source="openbookqa",
                difficulty=DifficultyLevel.ADVANCED, category="science_facts"))
        return out

    def _load_ms_marco(self, cap, hf_load):
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
            out.append(QAExample(
                question=q, context=ctx, answer=str(answer),
                id=f"msmarco_{i}", source="ms_marco",
                difficulty=DifficultyLevel.INTERMEDIATE, category="web_search_qa"))
        return out

    def _load_coqa(self, cap, hf_load):
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
            out.append(QAExample(
                question=str(q), context=story[:1500], answer=str(answer),
                id=f"coqa_{i}", source="coqa",
                difficulty=DifficultyLevel.INTERMEDIATE, category="conversational_qa"))
        return out

    # ── Sampling ──────────────────────────────────────────────────────────────

    def get_example_by_difficulty(self, difficulty: DifficultyLevel,
                                   exclude_used: bool = True) -> Optional[QAExample]:
        indices = self.indices_by_difficulty.get(difficulty, [])
        available = [i for i in indices if i not in self.used_indices] if exclude_used else list(indices)
        if not available:
            for diff in [DifficultyLevel.INTERMEDIATE, DifficultyLevel.BEGINNER,
                         DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:
                if diff != difficulty:
                    fb = self.indices_by_difficulty.get(diff, [])
                    available = [i for i in fb if i not in self.used_indices] if exclude_used else list(fb)
                    if available: break
        if not available: return None
        idx = random.choice(available)
        self.used_indices.add(idx)
        return self.examples[idx]

    def get_random_example(self, difficulty: Optional[DifficultyLevel] = None) -> Optional[QAExample]:
        if difficulty: return self.get_example_by_difficulty(difficulty)
        if not self.examples: return None
        available = [i for i in range(len(self.examples)) if i not in self.used_indices]
        if not available:
            self.used_indices.clear()
            available = list(range(len(self.examples)))
        idx = random.choice(available)
        self.used_indices.add(idx)
        return self.examples[idx]

    def start_new_episode(self, num_questions: int = 10,
                          difficulty: Optional[DifficultyLevel] = None,
                          category: Optional[str] = None,
                          mix_difficulties: bool = False) -> List[QAExample]:
        self.current_episode_examples = []
        if mix_difficulties:
            for diff in ([DifficultyLevel.BEGINNER]*2 + [DifficultyLevel.INTERMEDIATE]*3 +
                         [DifficultyLevel.ADVANCED]*3 + [DifficultyLevel.EXPERT]*2)[:num_questions]:
                ex = self.get_example_by_difficulty(diff)
                if ex: self.current_episode_examples.append(ex)
        elif difficulty:
            for _ in range(num_questions):
                ex = self.get_example_by_difficulty(difficulty)
                if ex: self.current_episode_examples.append(ex)
        else:
            for _ in range(num_questions):
                ex = self.get_random_example()
                if ex: self.current_episode_examples.append(ex)
        while len(self.current_episode_examples) < num_questions:
            ex = self.get_random_example()
            if ex: self.current_episode_examples.append(ex)
            else: break
        return self.current_episode_examples

    def get_example_for_step(self, step: int) -> Optional[QAExample]:
        if 0 <= step < len(self.current_episode_examples):
            return self.current_episode_examples[step]
        return None

    def load_from_json(self, filepath: str) -> int:
        initial = len(self.examples)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                try: diff = DifficultyLevel(item.get("difficulty", "intermediate"))
                except ValueError: diff = DifficultyLevel.INTERMEDIATE
                self.examples.append(QAExample(
                    question=item.get("question", ""), context=item.get("context", ""),
                    answer=item.get("answer", ""), id=item.get("id", str(len(self.examples))),
                    source=item.get("source", "custom"), difficulty=diff,
                    category=item.get("category", "general"),
                    entities=item.get("entities", []), metadata=item.get("metadata", {})))
            self._update_statistics()
            self._build_indices()
            return len(self.examples) - initial
        except Exception as e:
            print(f"load_from_json error: {e}")
            return 0

    def get_statistics(self) -> DatasetStatistics: return self.statistics
    def get_total_examples(self) -> int: return len(self.examples)
    def reset_usage(self) -> None: self.used_indices.clear()


    def _load_nq_open(self, cap, hf_load):
        ds = hf_load("nq_open", split="train[:%d]" % cap)
        out = []
        for i, item in enumerate(ds):
            q = item.get("question", "")
            answers = item.get("answer", [])
            answer = answers[0] if answers else ""
            if not q or not answer:
                continue
            out.append(QAExample(
                question=q,
                context="Answer based on your knowledge: " + q,
                answer=str(answer),
                id="nq_open_%d" % i,
                source="nq_open",
                difficulty=DifficultyLevel.INTERMEDIATE,
                category="open_domain_qa"))
        return out

    def _load_commonsense_qa(self, cap, hf_load):
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
            out.append(QAExample(
                question=q, context=ctx, answer=answer,
                id="csqa_%d" % i, source="commonsense_qa",
                difficulty=DifficultyLevel.INTERMEDIATE,
                category="commonsense_reasoning"))
        return out

    def _load_winogrande(self, cap, hf_load):
        ds = hf_load("allenai/winogrande", "winogrande_xl",
                     split="train[:%d]" % cap)
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
            out.append(QAExample(
                question="Which option correctly fills the blank? " + sentence,
                context=ctx, answer=answer,
                id="winogrande_%d" % i, source="winogrande",
                difficulty=DifficultyLevel.INTERMEDIATE,
                category="commonsense_reasoning"))
        return out


    # ── New v4.0 Dataset Loaders ──────────────────────────────────────────────

    def _load_squad_v2(self, cap, hf_load):
        ds = hf_load("rajpurkar/squad_v2", split=f"train[:{cap}]")
        out = []
        for i, item in enumerate(ds):
            ans = item.get("answers", {}).get("text", [])
            answer = ans[0] if ans else "No answer"
            ctx = item.get("context", "")
            if not ctx: continue
            out.append(QAExample(
                question=item["question"], context=ctx[:1500],
                answer=answer, id=f"squadv2_{i}", source="squad_v2",
                difficulty=DifficultyLevel.ADVANCED, category="reading_comprehension_unanswerable"))
        return out

    def _load_drop(self, cap, hf_load):
        try:
            ds = hf_load("ucinlp/drop", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                passage = item.get("passage", "")
                answers = item.get("answers_spans", {})
                spans = answers.get("spans", []) if isinstance(answers, dict) else []
                answer = spans[0] if spans else ""
                if not q or not passage or not answer: continue
                out.append(QAExample(
                    question=q, context=passage[:1500], answer=str(answer),
                    id=f"drop_{i}", source="drop",
                    difficulty=DifficultyLevel.EXPERT, category="numerical_reasoning"))
            return out
        except Exception as e:
            print(f"  drop loader error: {e}"); return []

    def _load_race(self, cap, hf_load):
        out = []
        try:
            for split in ["train", "validation", "test"]:
                ds = hf_load("ehovy/race", "all", split=split)
                for item in ds:
                    if len(out) >= cap: break
                    q = item.get("question", "")
                    article = item.get("article", "")
                    options = item.get("options", [])
                    ans_key = item.get("answer", "")
                    key_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    idx = key_map.get(ans_key, -1)
                    answer = options[idx] if 0 <= idx < len(options) else ""
                    if not q or not article or not answer: continue
                    out.append(QAExample(
                        question=q, context=article[:1500], answer=answer,
                        id=f"race_{len(out)}", source="race",
                        difficulty=DifficultyLevel.ADVANCED, category="reading_comprehension_exam"))
        except Exception as e:
            print(f"  race loader error: {e}")
        return out

    def _load_newsqa(self, cap, hf_load):
        try:
            ds = hf_load("lucadiliello/newsqa", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                story = item.get("story_text", item.get("context", ""))
                answers = item.get("answers", {})
                if isinstance(answers, list) and answers:
                    answer = str(answers[0])
                elif isinstance(answers, dict):
                    answer = str(answers.get("answer_token_ranges", ""))
                else:
                    answer = ""
                if not q or not story or not answer: continue
                out.append(QAExample(
                    question=str(q), context=str(story)[:1500], answer=answer,
                    id=f"newsqa_{i}", source="newsqa",
                    difficulty=DifficultyLevel.INTERMEDIATE, category="news_qa"))
            return out
        except Exception as e:
            print(f"  newsqa loader error: {e}"); return []

    def _load_hellaswag(self, cap, hf_load):
        try:
            ds = hf_load("Rowan/hellaswag", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                ctx = item.get("ctx", "")
                endings = item.get("endings", [])
                label = item.get("label", "")
                try:
                    idx = int(label)
                    answer = endings[idx] if 0 <= idx < len(endings) else ""
                except (ValueError, TypeError):
                    answer = ""
                if not ctx or not answer: continue
                choices_str = " | ".join(f"{j}: {e}" for j, e in enumerate(endings))
                out.append(QAExample(
                    question=f"What is the most likely continuation? {ctx}",
                    context=f"Context: {ctx} | Choices: {choices_str}",
                    answer=answer, id=f"hellaswag_{i}", source="hellaswag",
                    difficulty=DifficultyLevel.INTERMEDIATE, category="commonsense_completion"))
            return out
        except Exception as e:
            print(f"  hellaswag loader error: {e}"); return []

    def _load_adversarial_qa(self, cap, hf_load):
        try:
            ds = hf_load("adversarial_qa", "adversarialQA", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                ctx = item.get("context", "")
                ans = item.get("answers", {}).get("text", [])
                answer = ans[0] if ans else ""
                if not q or not ctx or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"advqa_{i}", source="adversarial_qa",
                    difficulty=DifficultyLevel.EXPERT, category="adversarial_reading_comprehension"))
            return out
        except Exception as e:
            print(f"  adversarial_qa loader error: {e}"); return []

    def _load_ag_news(self, cap, hf_load):
        try:
            ds = hf_load("fancyzhx/ag_news", split=f"train[:{cap}]")
            label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
            out = []
            for i, item in enumerate(ds):
                text = item.get("text", "")
                label = label_map.get(item.get("label", -1), "")
                if not text or not label: continue
                out.append(QAExample(
                    question="What is the topic category of this news article?",
                    context=text[:1500], answer=label,
                    id=f"agnews_{i}", source="ag_news",
                    difficulty=DifficultyLevel.BEGINNER, category="news_classification"))
            return out
        except Exception as e:
            print(f"  ag_news loader error: {e}"); return []

    def _load_aqua_rat(self, cap, hf_load):
        try:
            ds = hf_load("aqua_rat", "raw", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                options = item.get("options", [])
                correct = item.get("correct", "")
                rationale = item.get("rationale", "")
                answer = next((o for o in options if o.startswith(correct + ")")), correct)
                ctx = f"Options: {' | '.join(options)} | Rationale: {rationale}"
                if not q or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"aquarat_{i}", source="aqua_rat",
                    difficulty=DifficultyLevel.EXPERT, category="math_word_problems"))
            return out
        except Exception as e:
            print(f"  aqua_rat loader error: {e}"); return []

    def _load_circa(self, cap, hf_load):
        try:
            ds = hf_load("circa", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question_x", "")
                ans = item.get("answer_y", "")
                ctx = item.get("context", item.get("canquestion_x", q))
                judgement = item.get("goldstandard1", "")
                if not q or not ans: continue
                out.append(QAExample(
                    question=q, context=str(ctx)[:1500], answer=str(ans),
                    id=f"circa_{i}", source="circa",
                    difficulty=DifficultyLevel.INTERMEDIATE, category="social_context_qa"))
            return out
        except Exception as e:
            print(f"  circa loader error: {e}"); return []

    def _load_climate_fever(self, cap, hf_load):
        try:
            ds = hf_load("climate_fever", split=f"test[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                claim = item.get("claim", "")
                label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO", 3: "DISPUTED"}
                label = label_map.get(item.get("claim_label", 2), "NOT ENOUGH INFO")
                evidences = item.get("evidences", [])
                ctx = " ".join([e.get("evidence", "") for e in evidences[:3]])[:1500] if evidences else claim
                if not claim: continue
                out.append(QAExample(
                    question=f"Is this climate claim supported? Claim: {claim}",
                    context=ctx, answer=label,
                    id=f"climatefever_{i}", source="climate_fever",
                    difficulty=DifficultyLevel.EXPERT, category="climate_fact_verification"))
            return out
        except Exception as e:
            print(f"  climate_fever loader error: {e}"); return []

    def _load_cnn_dailymail(self, cap, hf_load):
        try:
            ds = hf_load("abisee/cnn_dailymail", "3.0.0", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                article = item.get("article", "")
                highlights = item.get("highlights", "")
                if not article or not highlights: continue
                out.append(QAExample(
                    question="What are the key points of this article?",
                    context=article[:1500], answer=highlights[:500],
                    id=f"cnndm_{i}", source="cnn_dailymail",
                    difficulty=DifficultyLevel.INTERMEDIATE, category="news_summarisation"))
            return out
        except Exception as e:
            print(f"  cnn_dailymail loader error: {e}"); return []

    def _load_scitail(self, cap, hf_load):
        try:
            ds = hf_load("allenai/scitail", "tsv_format", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                premise    = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                label      = item.get("label", "neutral")
                answer     = "SUPPORTS" if label == "entails" else "REFUTES"
                if not premise or not hypothesis: continue
                out.append(QAExample(
                    question=f"Does the premise support this hypothesis? Hypothesis: {hypothesis}",
                    context=f"Premise: {premise}", answer=answer,
                    id=f"scitail_{i}", source="scitail",
                    difficulty=DifficultyLevel.ADVANCED, category="science_entailment"))
            return out
        except Exception as e:
            print(f"  scitail loader error: {e}"); return []

    def _load_medqa(self, cap, hf_load):
        try:
            ds = hf_load("GBaker/MedQA-USMLE-4-options", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                options = item.get("options", {})
                answer_idx = item.get("answer_idx", "")
                answer = options.get(answer_idx, "") if isinstance(options, dict) else ""
                ctx = "Options: " + " | ".join(f"{k}: {v}" for k, v in options.items()) if isinstance(options, dict) else ""
                if not q or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"medqa_{i}", source="medqa",
                    difficulty=DifficultyLevel.EXPERT, category="medical_qa"))
            return out
        except Exception as e:
            print(f"  medqa loader error: {e}"); return []

    def _load_medmcqa(self, cap, hf_load):
        try:
            ds = hf_load("openlifescienceai/medmcqa", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                opts = [item.get(k, "") for k in ["opa", "opb", "opc", "opd"]]
                cop = item.get("cop", 0)
                answer = opts[cop - 1] if 1 <= cop <= 4 else ""
                ctx = "Options: " + " | ".join(f"{chr(65+j)}: {o}" for j, o in enumerate(opts))
                if not q or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"medmcqa_{i}", source="medmcqa",
                    difficulty=DifficultyLevel.EXPERT, category="medical_mcq"))
            return out
        except Exception as e:
            print(f"  medmcqa loader error: {e}"); return []

    def _load_medical_questions(self, cap, hf_load):
        try:
            ds = hf_load("medical_questions_pairs", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q1 = item.get("question_1", "")
                q2 = item.get("question_2", "")
                label = item.get("label", 0)
                answer = "similar" if label == 1 else "different"
                ctx = f"Question 1: {q1} | Question 2: {q2}"
                if not q1 or not q2: continue
                out.append(QAExample(
                    question=f"Are these two medical questions asking about the same thing?",
                    context=ctx[:1500], answer=answer,
                    id=f"medpairs_{i}", source="medical_questions",
                    difficulty=DifficultyLevel.ADVANCED, category="medical_similarity"))
            return out
        except Exception as e:
            print(f"  medical_questions loader error: {e}"); return []

    def _load_qasc(self, cap, hf_load):
        try:
            ds = hf_load("allenai/qasc", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                choices = item.get("choices", {})
                labels = choices.get("label", []) if isinstance(choices, dict) else []
                texts  = choices.get("text", []) if isinstance(choices, dict) else []
                ans_key = item.get("answerKey", "")
                fact1 = item.get("fact1", "")
                fact2 = item.get("fact2", "")
                answer = next((t for l, t in zip(labels, texts) if l == ans_key), "")
                ctx = f"Fact 1: {fact1} | Fact 2: {fact2} | Choices: " + " | ".join(f"{l}: {t}" for l, t in zip(labels, texts))
                if not q or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"qasc_{i}", source="qasc",
                    difficulty=DifficultyLevel.ADVANCED, category="multi_hop_science"))
            return out
        except Exception as e:
            print(f"  qasc loader error: {e}"); return []

    def _load_quartz(self, cap, hf_load):
        try:
            ds = hf_load("allenai/quartz", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                choices = item.get("choices", {})
                labels = choices.get("label", []) if isinstance(choices, dict) else []
                texts  = choices.get("text", []) if isinstance(choices, dict) else []
                ans_key = item.get("answerKey", "")
                para = item.get("para", "")
                answer = next((t for l, t in zip(labels, texts) if l == ans_key), "")
                ctx = f"{para} | Choices: " + " | ".join(f"{l}: {t}" for l, t in zip(labels, texts))
                if not q or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"quartz_{i}", source="quartz",
                    difficulty=DifficultyLevel.ADVANCED, category="qualitative_science"))
            return out
        except Exception as e:
            print(f"  quartz loader error: {e}"); return []

    def _load_quail(self, cap, hf_load):
        try:
            ds = hf_load("potsawee/quail", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                ctx = item.get("context", "")
                answers = item.get("answers", [])
                correct_idx = item.get("correct_answer_id", 0)
                answer = answers[correct_idx] if answers and 0 <= correct_idx < len(answers) else ""
                if not q or not ctx or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"quail_{i}", source="quail",
                    difficulty=DifficultyLevel.ADVANCED, category="reading_comprehension"))
            return out
        except Exception as e:
            print(f"  quail loader error: {e}"); return []

    def _load_pubmedqa(self, cap, hf_load):
        try:
            ds = hf_load("qiaojin/PubMedQA", "pqa_labeled", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                ctx_list = item.get("context", {})
                if isinstance(ctx_list, dict):
                    ctx = " ".join(ctx_list.get("contexts", []))[:1500]
                else:
                    ctx = str(ctx_list)[:1500]
                answer = item.get("long_answer", item.get("final_decision", ""))
                if not q or not ctx or not answer: continue
                out.append(QAExample(
                    question=q, context=ctx, answer=str(answer),
                    id=f"pubmedqa_{i}", source="pubmedqa",
                    difficulty=DifficultyLevel.EXPERT, category="biomedical_qa"))
            return out
        except Exception as e:
            print(f"  pubmedqa loader error: {e}"); return []

    def _load_xsum(self, cap, hf_load):
        try:
            ds = hf_load("EdinburghNLP/xsum", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                doc = item.get("document", "")
                summary = item.get("summary", "")
                if not doc or not summary: continue
                out.append(QAExample(
                    question="Summarise this document in one sentence.",
                    context=doc[:1500], answer=summary,
                    id=f"xsum_{i}", source="xsum",
                    difficulty=DifficultyLevel.ADVANCED, category="summarisation"))
            return out
        except Exception as e:
            print(f"  xsum loader error: {e}"); return []

    def _load_sciq(self, cap, hf_load):
        try:
            ds = hf_load("allenai/sciq", split=f"train[:{cap}]")
            out = []
            for i, item in enumerate(ds):
                q = item.get("question", "")
                support = item.get("support", "")
                answer = item.get("correct_answer", "")
                d1 = item.get("distractor1", "")
                d2 = item.get("distractor2", "")
                d3 = item.get("distractor3", "")
                if not q or not answer: continue
                ctx = f"{support} | Options: {answer} | {d1} | {d2} | {d3}" if support else                       f"Options: {answer} | {d1} | {d2} | {d3}"
                out.append(QAExample(
                    question=q, context=ctx[:1500], answer=answer,
                    id=f"sciq_{i}", source="sciq",
                    difficulty=DifficultyLevel.ADVANCED, category="science_qa"))
            return out
        except Exception as e:
            print(f"  sciq loader error: {e}"); return []

    def _update_statistics(self) -> None:
        self.statistics.total_examples = len(self.examples)
        self.statistics.examples_by_source = {}
        self.statistics.examples_by_difficulty = {}
        self.statistics.examples_by_category = {}
        for ex in self.examples:
            for d, k in [(self.statistics.examples_by_source, ex.source),
                         (self.statistics.examples_by_difficulty, ex.difficulty.value),
                         (self.statistics.examples_by_category, ex.category)]:
                d[k] = d.get(k, 0) + 1
        if self.examples:
            self.statistics.average_context_length = sum(len(e.context) for e in self.examples) / len(self.examples)
            self.statistics.average_question_length = sum(len(e.question) for e in self.examples) / len(self.examples)

    def _build_indices(self) -> None:
        self.indices_by_difficulty = {d: [] for d in DifficultyLevel}
        self.indices_by_category = {}
        for i, ex in enumerate(self.examples):
            self.indices_by_difficulty[ex.difficulty].append(i)
            if ex.category not in self.indices_by_category:
                self.indices_by_category[ex.category] = []
            self.indices_by_category[ex.category].append(i)
