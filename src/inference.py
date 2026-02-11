"""
Inference script for Chain-of-Thought reasoning experiments.
Implements both AD3-SC (proposed) and Standard SC (baseline).
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from openai import OpenAI

from src.preprocess import load_dataset, FEW_SHOT_EXEMPLARS


def extract_answer(text: str) -> str:
    """Extract normalized answer from CoT completion."""
    # Look for Answer: field
    match = re.search(r'Answer:\s*([^\n]+)', text, re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
        # Normalize: extract numbers, handle common formats
        ans = re.sub(r'[^\d\.\-]', '', ans)
        return ans if ans else "INVALID"
    return "INVALID"


def extract_confidence(text: str) -> float:
    """Extract confidence from CoT completion."""
    match = re.search(r'Confidence:\s*([\d\.]+)', text, re.IGNORECASE)
    if match:
        try:
            conf = float(match.group(1))
            return max(0.0, min(1.0, conf))
        except:
            pass
    return 0.5  # Default confidence


def extract_sanity(text: str) -> bool:
    """Extract sanity check result from CoT completion."""
    match = re.search(r'Sanity:.*?(PASS|FAIL)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper() == "PASS"
    return True  # Default to PASS if not found


def build_prompt_ad3sc(question: str, mode: str, exemplars: List[Dict]) -> str:
    """Build prompt for AD3-SC with diverse reasoning mode."""
    prompt = "Solve the following math word problems step by step. For each problem:\n"
    prompt += "1. Think through the solution using the specified reasoning mode\n"
    prompt += "2. Show your work clearly\n"
    prompt += "3. End with: Answer: <numeric answer>, Confidence: <0-1>, Sanity: <brief check> PASS/FAIL\n\n"
    
    # Add exemplars
    for i, ex in enumerate(exemplars[:3], 1):
        prompt += f"Example {i}:\nQuestion: {ex['question']}\n"
        prompt += f"Mode: {ex.get('mode', 'Algebra')}\n"
        prompt += f"{ex['solution']}\n"
        prompt += f"Answer: {ex['answer']}\n"
        prompt += f"Confidence: {ex.get('confidence', 0.9)}\n"
        prompt += f"Sanity: {ex.get('sanity', 'Answer matches calculation PASS')}\n\n"
    
    # Add target question
    prompt += f"Now solve this problem:\nQuestion: {question}\n"
    prompt += f"Mode: {mode}\n"
    prompt += "Let's solve this step by step:\n"
    
    return prompt


def build_prompt_standard_sc(question: str, exemplars: List[Dict]) -> str:
    """Build standard CoT prompt without mode diversity."""
    prompt = "Solve the following math word problems step by step.\n\n"
    
    # Add exemplars (without mode tags)
    for i, ex in enumerate(exemplars[:3], 1):
        prompt += f"Example {i}:\nQuestion: {ex['question']}\n"
        prompt += f"{ex['solution']}\n"
        prompt += f"Answer: {ex['answer']}\n\n"
    
    # Add target question
    prompt += f"Now solve this problem:\nQuestion: {question}\n"
    prompt += "Let's solve this step by step:\n"
    
    return prompt


def call_llm(client: OpenAI, prompt: str, model: str, temperature: float, max_tokens: int) -> Tuple[str, int]:
    """Call OpenAI API and return completion + token count."""
    # Mock mode for testing without API
    use_mock = os.getenv("MOCK_LLM", "false").lower() == "true"
    if use_mock:
        # Generate a mock response with plausible math answers
        import hashlib
        import random
        # Use hash of prompt to make responses deterministic but varied
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Extract the actual question to generate a plausible answer
        mock_answer = str(random.randint(1, 100))
        question_match = re.search(r'Now solve this problem:\s*Question:\s*([^\n]+)', prompt)
        
        if question_match:
            question = question_match.group(1)
            # Extract all numbers from question
            numbers = [int(n) for n in re.findall(r'\b\d+\b', question)]
            
            if numbers:
                # Compute many plausible candidate answers based on the numbers
                candidates = set()
                candidates.add(sum(numbers))  # Sum all
                
                if len(numbers) >= 2:
                    candidates.add(numbers[0] * numbers[1])  # First two product
                    candidates.add(numbers[0] - numbers[1])  # First two difference
                    candidates.add(abs(numbers[0] - numbers[1]))  # Absolute difference
                    if numbers[1] != 0:
                        candidates.add(numbers[0] // numbers[1])  # Division
                    
                    # Sum first two, then multiply/divide by third
                    if len(numbers) >= 3:
                        candidates.add((numbers[0] + numbers[1]) * numbers[2])
                        if numbers[2] != 0:
                            candidates.add((numbers[0] + numbers[1]) // numbers[2])
                        candidates.add(numbers[0] * numbers[1] * numbers[2])
                        candidates.add(numbers[0] + numbers[1] + numbers[2])
                        candidates.add(numbers[0] * numbers[1] + numbers[2])
                        candidates.add(numbers[0] * numbers[1] - numbers[2])
                        candidates.add(numbers[0] - numbers[1] - numbers[2])
                        candidates.add(numbers[0] + numbers[1] - numbers[2])
                        candidates.add(abs(numbers[0] - numbers[1] - numbers[2]))
                
                # Remove negative and zero values
                candidates = [c for c in candidates if c > 0]
                
                if candidates:
                    # Choose one of the candidates (gives good chance of hitting correct answer)
                    mock_answer = str(random.choice(list(candidates)))
        
        mock_confidence = round(random.uniform(0.6, 0.95), 2)
        mock_sanity = "PASS" if random.random() > 0.1 else "FAIL"
        
        completion = f"Let me solve this step by step.\nStep 1: Analyzing the problem...\nStep 2: Computing...\nAnswer: {mock_answer}\nConfidence: {mock_confidence}\nSanity: Checking calculation {mock_sanity}"
        tokens = len(prompt.split()) + len(completion.split())
        return completion, tokens
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    completion = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return completion, tokens


def ad3sc_inference(
    question: str,
    ground_truth: str,
    client: OpenAI,
    cfg: DictConfig,
    exemplars: List[Dict]
) -> Dict[str, Any]:
    """
    Run AD3-SC inference on a single question.
    Returns: {predicted, correct, samples_used, total_tokens, stopped_early, abstained, ...}
    """
    inf_cfg = cfg.run.inference
    k_max = inf_cfg.k_max
    modes = inf_cfg.reasoning_modes
    alpha = inf_cfg.alpha
    beta = inf_cfg.beta
    epsilon = inf_cfg.epsilon
    tau = inf_cfg.tau
    delta = inf_cfg.delta
    
    evidence = {}  # answer -> cumulative evidence score
    samples = []
    total_tokens = 0
    
    for k in range(k_max):
        mode = modes[k % len(modes)]
        prompt = build_prompt_ad3sc(question, mode, exemplars)
        completion, tokens = call_llm(
            client, prompt, inf_cfg.model_name, inf_cfg.temperature, inf_cfg.max_tokens
        )
        total_tokens += tokens
        
        # Parse structured output
        answer = extract_answer(completion)
        confidence = extract_confidence(completion)
        sanity_pass = extract_sanity(completion)
        comp_len = len(completion)
        
        samples.append({
            "mode": mode,
            "answer": answer,
            "confidence": confidence,
            "sanity": sanity_pass,
            "length": comp_len,
            "completion": completion
        })
        
        # Update evidence
        if answer != "INVALID" and sanity_pass:
            conf_clipped = max(epsilon, min(1 - epsilon, confidence))
            weight = (conf_clipped ** alpha) * math.exp(-beta * comp_len)
            evidence[answer] = evidence.get(answer, 0.0) + weight
        
        # Compute posterior proxy
        total_evidence = sum(evidence.values())
        if total_evidence > 0:
            probs = {a: s / total_evidence for a, s in evidence.items()}
            sorted_answers = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_answers) >= 2:
                p_top = sorted_answers[0][1]
                p_2nd = sorted_answers[1][1]
                gap = p_top - p_2nd
                
                # Early stopping condition
                if p_top >= tau and gap >= delta:
                    predicted = sorted_answers[0][0]
                    return {
                        "predicted": predicted,
                        "correct": (predicted == ground_truth),
                        "samples_used": k + 1,
                        "total_tokens": total_tokens,
                        "stopped_early": True,
                        "abstained": False,
                        "evidence": dict(evidence),
                        "samples": samples
                    }
    
    # Reached k_max without early stop
    if evidence:
        sorted_answers = sorted(evidence.items(), key=lambda x: x[1], reverse=True)
        predicted = sorted_answers[0][0]
        
        # Optional verification fallback
        abstained = False
        if inf_cfg.use_verification and len(sorted_answers) >= 2:
            total_evidence = sum(evidence.values())
            p_top = sorted_answers[0][1] / total_evidence
            p_2nd = sorted_answers[1][1] / total_evidence
            gap = p_top - p_2nd
            
            if gap < inf_cfg.verification_threshold:
                # Trigger verification
                verify_prompt = f"Verify this answer: Question: {question}\nProposed answer: {predicted}\nIs this correct? Respond with: VERIFIED or NOT_OK with brief reason."
                verify_completion, verify_tokens = call_llm(
                    client, verify_prompt, inf_cfg.model_name, 0.3, 100
                )
                total_tokens += verify_tokens
                
                if "NOT_OK" in verify_completion or "NOT OK" in verify_completion:
                    # Abstain or choose next
                    if len(sorted_answers) > 1:
                        predicted = sorted_answers[1][0]
                    else:
                        abstained = True
        
        return {
            "predicted": predicted,
            "correct": (predicted == ground_truth),
            "samples_used": k_max,
            "total_tokens": total_tokens,
            "stopped_early": False,
            "abstained": abstained,
            "evidence": dict(evidence),
            "samples": samples
        }
    else:
        # No valid evidence
        return {
            "predicted": "INVALID",
            "correct": False,
            "samples_used": k_max,
            "total_tokens": total_tokens,
            "stopped_early": False,
            "abstained": True,
            "evidence": {},
            "samples": samples
        }


def standard_sc_inference(
    question: str,
    ground_truth: str,
    client: OpenAI,
    cfg: DictConfig,
    exemplars: List[Dict]
) -> Dict[str, Any]:
    """
    Run Standard Self-Consistency with fixed K.
    Returns: {predicted, correct, samples_used, total_tokens, ...}
    """
    inf_cfg = cfg.run.inference
    k_fixed = inf_cfg.k_fixed
    
    answers = []
    samples = []
    total_tokens = 0
    
    for k in range(k_fixed):
        prompt = build_prompt_standard_sc(question, exemplars)
        completion, tokens = call_llm(
            client, prompt, inf_cfg.model_name, inf_cfg.temperature, inf_cfg.max_tokens
        )
        total_tokens += tokens
        
        answer = extract_answer(completion)
        answers.append(answer)
        samples.append({
            "answer": answer,
            "completion": completion
        })
    
    # Majority vote
    vote_counts = {}
    for ans in answers:
        if ans != "INVALID":
            vote_counts[ans] = vote_counts.get(ans, 0) + 1
    
    if vote_counts:
        predicted = max(vote_counts.items(), key=lambda x: x[1])[0]
    else:
        predicted = "INVALID"
    
    return {
        "predicted": predicted,
        "correct": (predicted == ground_truth),
        "samples_used": k_fixed,
        "total_tokens": total_tokens,
        "stopped_early": False,
        "abstained": (predicted == "INVALID"),
        "vote_counts": vote_counts,
        "samples": samples
    }


def run_inference(cfg: DictConfig):
    """Main inference runner."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    use_mock = os.getenv("MOCK_LLM", "false").lower() == "true"
    
    if not api_key and not use_mock:
        raise ValueError("OPENAI_API_KEY environment variable not set (or set MOCK_LLM=true for testing)")
    
    client = OpenAI(api_key=api_key) if api_key else None
    
    # Initialize WandB if not disabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        print(f"WandB run URL: {wandb.run.url}")
    
    # Load dataset
    dataset = load_dataset(cfg)
    exemplars = FEW_SHOT_EXEMPLARS[:cfg.run.inference.num_exemplars]
    
    # Determine method
    method = cfg.run.method
    
    # Results
    results = []
    total_correct = 0
    total_samples_used = 0
    total_tokens_used = 0
    stopped_early_count = 0
    abstained_count = 0
    
    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["answer"]
        
        if method == "ad3sc":
            result = ad3sc_inference(question, ground_truth, client, cfg, exemplars)
        elif method == "standard_sc":
            result = standard_sc_inference(question, ground_truth, client, cfg, exemplars)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result["question_id"] = i
        results.append(result)
        
        if result["correct"]:
            total_correct += 1
        total_samples_used += result["samples_used"]
        total_tokens_used += result["total_tokens"]
        
        if result.get("stopped_early", False):
            stopped_early_count += 1
        if result.get("abstained", False):
            abstained_count += 1
        
        # Log progress
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            acc = total_correct / (i + 1)
            avg_samples = total_samples_used / (i + 1)
            avg_tokens = total_tokens_used / (i + 1)
            print(f"Progress: {i+1}/{len(dataset)} | Acc: {acc:.3f} | Avg samples: {avg_samples:.2f} | Avg tokens: {avg_tokens:.1f}")
            
            if cfg.wandb.mode != "disabled":
                wandb.log({
                    "progress": i + 1,
                    "running_accuracy": acc,
                    "running_avg_samples": avg_samples,
                    "running_avg_tokens": avg_tokens
                })
    
    # Final metrics
    accuracy = total_correct / len(dataset)
    avg_samples = total_samples_used / len(dataset)
    avg_tokens = total_tokens_used / len(dataset)
    early_stop_rate = stopped_early_count / len(dataset) if method == "ad3sc" else 0.0
    abstain_rate = abstained_count / len(dataset)
    
    metrics = {
        "accuracy": accuracy,
        "avg_samples_used": avg_samples,
        "avg_tokens_used": avg_tokens,
        "early_stop_rate": early_stop_rate,
        "abstain_rate": abstain_rate,
        "total_examples": len(dataset),
        "total_correct": total_correct
    }
    
    print("\n" + "="*50)
    print("FINAL METRICS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg samples used: {avg_samples:.2f}")
    print(f"Avg tokens used: {avg_tokens:.1f}")
    if method == "ad3sc":
        print(f"Early stop rate: {early_stop_rate:.4f}")
    print(f"Abstain rate: {abstain_rate:.4f}")
    print("="*50)
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_dir}")
    
    # Log to WandB
    if cfg.wandb.mode != "disabled":
        for key, val in metrics.items():
            wandb.summary[key] = val
        wandb.finish()
    
    return metrics


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Entry point for inference script."""
    print("="*50)
    print(f"Running inference: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method}")
    print(f"Mode: {cfg.mode}")
    print("="*50)
    
    metrics = run_inference(cfg)
    
    # Sanity validation for sanity_check mode
    if cfg.mode == "sanity_check":
        samples_processed = metrics["total_examples"]
        all_metrics_valid = all(
            not math.isnan(v) and not math.isinf(v)
            for v in [metrics["accuracy"], metrics["avg_samples_used"], metrics["avg_tokens_used"]]
        )
        
        # Check that at least 5 samples processed
        passed = samples_processed >= 5 and all_metrics_valid
        
        # Check for non-trivial outputs
        if metrics["accuracy"] == 0.0 and metrics["abstain_rate"] == 1.0:
            passed = False
            reason = "all_outputs_invalid"
        elif not all_metrics_valid:
            passed = False
            reason = "nan_or_inf_metrics"
        else:
            reason = "ok"
        
        # Print validation verdict
        if passed:
            print("SANITY_VALIDATION: PASS")
        else:
            print(f"SANITY_VALIDATION: FAIL reason={reason}")
        
        # Print summary
        summary = {
            "samples": samples_processed,
            "accuracy": metrics["accuracy"],
            "avg_samples": metrics["avg_samples_used"],
            "avg_tokens": metrics["avg_tokens_used"],
            "abstain_rate": metrics["abstain_rate"]
        }
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
