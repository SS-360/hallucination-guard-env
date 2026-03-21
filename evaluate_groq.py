"""
HallucinationGuard-Env — Groq/Llama Evaluator (SDK version)
Uses the HallucinationGuard SDK + Groq free tier

Setup:
    pip install groq requests
    Get free key at https://console.groq.com
    python evaluate_groq.py --api-key YOUR_GROQ_KEY --episodes 5
"""

import argparse
import sys

try:
    from groq import Groq
except ImportError:
    print("Run: pip install groq requests")
    sys.exit(1)

from hallucination_guard_sdk import HallucinationGuardEnv

MODEL = "llama-3.1-8b-instant"

SYSTEM = """Answer questions using ONLY the provided context.
If the context lacks real information, say: "The context does not contain enough information."
Never use outside knowledge. Be concise."""

def make_model_fn(client):
    def model_fn(question: str, context: str) -> str:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        return r.choices[0].message.content.strip()
    return model_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",      required=True)
    parser.add_argument("--episodes",     type=int, default=5)
    parser.add_argument("--model-name",   default="llama-3.1-8b-groq")
    parser.add_argument("--organization", default="")
    parser.add_argument("--submit",       action="store_true",
                        help="Submit results to leaderboard")
    args = parser.parse_args()

    client   = Groq(api_key=args.api_key)
    model_fn = make_model_fn(client)

    env     = HallucinationGuardEnv()
    results = env.evaluate(model_fn, episodes=args.episodes,
                           model_name=args.model_name)
    env.save_results(results)

    if args.submit:
        env.submit_to_leaderboard(results, organization=args.organization)

if __name__ == "__main__":
    main()
