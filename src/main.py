"""
Main orchestrator for running inference experiments.
Loads config and invokes inference.py as a subprocess.
"""

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main entry point for running experiments.
    Orchestrates inference runs based on config.
    """
    print("="*60)
    print(f"Main orchestrator starting")
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method}")
    print(f"Dataset: {cfg.run.dataset}")
    print(f"Mode: {cfg.mode}")
    print("="*60)
    
    # Check if we need to use mock LLM mode (no API key available)
    import os
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["MOCK_LLM"] = "true"
        print(f"OPENAI_API_KEY not set, enabling MOCK_LLM mode for {cfg.mode} mode")
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        # Override for sanity check: limit dataset size and disable wandb
        OmegaConf.set_struct(cfg, False)
        cfg.wandb.mode = "disabled"
        cfg.run.dataset_params.max_samples = 10  # Process only 10 samples
        OmegaConf.set_struct(cfg, True)
        print(f"Sanity check mode: Processing max {cfg.run.dataset_params.max_samples} samples")
    
    # Determine task type and invoke appropriate script
    task = cfg.run.task
    
    if task == "inference":
        # Run inference.py directly (not as subprocess since it uses @hydra.main)
        print("Running inference task...")
        from src.inference import run_inference
        
        try:
            metrics = run_inference(cfg)
            
            # Sanity validation for sanity_check mode
            if cfg.mode == "sanity_check":
                import json
                import math
                
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
        
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            
            if cfg.mode == "sanity_check":
                print(f"SANITY_VALIDATION: FAIL reason=exception")
                print(f"SANITY_VALIDATION_SUMMARY: {{'error': '{str(e)}'}}")
            sys.exit(1)
    
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    print("="*60)
    print("Main orchestrator completed successfully")
    print("="*60)


if __name__ == "__main__":
    main()
