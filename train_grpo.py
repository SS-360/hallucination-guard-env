"""
train_grpo.py — GRPO training with HallucinationGuard-Env
==========================================================
Trains Qwen3-1.7B to avoid hallucinations using TRL GRPOTrainer
connected to the live HallucinationGuard-Env OpenEnv environment.

Hardware : A100 40GB GPU (Colab Pro or equivalent)
Time     : ~90 minutes for 1 epoch

Install:
    pip install trl transformers torch openenv-halluguard datasets

Run:
    python train_grpo.py

Or in a Colab notebook:
    !python train_grpo.py
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL    = "Qwen/Qwen3-1.7B"
BASE_URL = "https://samsankar-hallucination-guard-env.hf.space"
OUTPUT   = "./hallucination-guard-grpo"

# ── Connect to live environment ────────────────────────────────────────────────

from openenv_halluguard import HallucinationGuardEnv

print("Connecting to HallucinationGuard-Env...")
env = HallucinationGuardEnv(base_url=BASE_URL, verbose=False)
print(f"Connected. Dataset: {env.dataset_info().get('total_examples', '?'):,} examples")

# ── Build prompt dataset ───────────────────────────────────────────────────────

def build_prompt_dataset(n_samples: int = 500) -> Dataset:
    """
    Sample questions from the environment to build the GRPO prompt dataset.
    Each prompt instructs the model to answer from context only.
    """
    prompts = []
    print(f"Sampling {n_samples} prompts from environment...")

    for i in range(n_samples):
        obs = env.reset()
        question = obs.get("question", "")
        context  = obs.get("context", "")

        prompt = (
            f"You are a factual question-answering assistant. "
            f"Answer the question using ONLY the information in the provided context. "
            f"Do not add any information not present in the context. "
            f"If the answer is not in the context, say 'I cannot answer from the provided context.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        prompts.append({"prompt": prompt, "question": question, "context": context})

        if (i + 1) % 50 == 0:
            print(f"  Sampled {i + 1}/{n_samples} prompts")

    return Dataset.from_list(prompts)


# ── Reward function ────────────────────────────────────────────────────────────

def hallucination_reward(prompts, completions, **kwargs):
    """
    Reward function that evaluates each completion against
    the HallucinationGuard-Env grader.

    Returns a list of reward floats in [0.0, 1.0] where:
      - 0.85-1.00 = grounded correct answer
      - 0.40-0.70 = partially grounded
      - 0.00-0.10 = hallucinated / fabricated answer
    """
    rewards = []

    for prompt, completion in zip(prompts, completions):
        try:
            obs    = env.reset()
            result = env.step(completion)

            reward          = float(result.get("reward", 0.0))
            is_hallucination = result.get("is_hallucination", False)
            severity        = result.get("hallucination_severity", "none")

            # Extra penalty for critical hallucinations
            if severity == "critical":
                reward = min(reward, 0.05)
            elif severity == "severe":
                reward = min(reward, 0.20)

            rewards.append(reward)

        except Exception as e:
            print(f"Reward eval error: {e}")
            rewards.append(0.0)

    return rewards


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    print(f"\nLoading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building prompt dataset...")
    dataset = build_prompt_dataset(n_samples=500)
    print(f"Dataset built: {len(dataset)} prompts")

    config = GRPOConfig(
        output_dir=OUTPUT,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        max_prompt_length=512,
        max_completion_length=256,
        num_generations=4,

        # vLLM colocate mode — shares GPU for generation + training
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.1,

        # Logging
        logging_steps=10,
        save_steps=100,
        report_to="none",

        # Reproducibility
        seed=42,
    )

    print(f"\nStarting GRPO training...")
    print(f"  Model   : {MODEL}")
    print(f"  Samples : {len(dataset)}")
    print(f"  Output  : {OUTPUT}")
    print(f"  Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=hallucination_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\nSaving model to {OUTPUT}-final")
    trainer.save_model(f"{OUTPUT}-final")
    tokenizer.save_pretrained(f"{OUTPUT}-final")

    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT}-final")
    print(f"Push to HF Hub:")
    print(f"  trainer.push_to_hub('your-username/hallucination-guard-qwen3')")


if __name__ == "__main__":
    main()
