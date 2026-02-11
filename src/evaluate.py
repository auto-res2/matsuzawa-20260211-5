"""
Evaluation script for comparing inference methods.
Fetches results from WandB and generates comparison metrics and figures.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs to compare")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (optional)")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project (optional)")
    return parser.parse_args()


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.
    
    Returns:
        Dictionary with 'config', 'summary', and 'history' keys
    """
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"
    
    try:
        run = api.run(run_path)
        
        # Get summary metrics
        summary = dict(run.summary)
        
        # Get config
        config = dict(run.config)
        
        # Get history (time series metrics)
        history = run.history()
        history_list = history.to_dict('records') if not history.empty else []
        
        return {
            "config": config,
            "summary": summary,
            "history": history_list,
            "url": run.url
        }
    except Exception as e:
        print(f"Warning: Could not fetch WandB run {run_id}: {e}")
        return {
            "config": {},
            "summary": {},
            "history": [],
            "url": None
        }


def load_local_metrics(results_dir: Path, run_id: str) -> Dict[str, Any]:
    """Load metrics from local results directory."""
    metrics_path = results_dir / run_id / "metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    else:
        print(f"Warning: Local metrics not found for {run_id}")
        return {}


def create_comparison_plots(run_data: Dict[str, Dict], results_dir: Path):
    """Create comparison plots for all runs."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    run_ids = list(run_data.keys())
    
    # Extract metrics for plotting
    accuracies = []
    avg_samples = []
    avg_tokens = []
    labels = []
    
    for run_id in run_ids:
        metrics = run_data[run_id].get("local_metrics", {})
        if not metrics:
            metrics = run_data[run_id].get("wandb_data", {}).get("summary", {})
        
        labels.append(run_id)
        accuracies.append(metrics.get("accuracy", 0.0))
        avg_samples.append(metrics.get("avg_samples_used", 0.0))
        avg_tokens.append(metrics.get("avg_tokens_used", 0.0))
    
    # Plot 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(labels)), accuracies, color=['#2E86AB', '#A23B72'])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    accuracy_plot_path = comparison_dir / "accuracy_comparison.png"
    plt.savefig(accuracy_plot_path, dpi=150)
    plt.close()
    print(f"Saved: {accuracy_plot_path}")
    
    # Plot 2: Average samples used
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(labels)), avg_samples, color=['#2E86AB', '#A23B72'])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Avg Samples Used')
    ax.set_title('Average Samples Used per Question')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    samples_plot_path = comparison_dir / "avg_samples_comparison.png"
    plt.savefig(samples_plot_path, dpi=150)
    plt.close()
    print(f"Saved: {samples_plot_path}")
    
    # Plot 3: Average tokens used
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(labels)), avg_tokens, color=['#2E86AB', '#A23B72'])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Avg Tokens Used')
    ax.set_title('Average Tokens Used per Question')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    tokens_plot_path = comparison_dir / "avg_tokens_comparison.png"
    plt.savefig(tokens_plot_path, dpi=150)
    plt.close()
    print(f"Saved: {tokens_plot_path}")
    
    # Plot 4: Efficiency scatter (accuracy vs compute)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(labels):
        ax.scatter(avg_samples[i], accuracies[i], s=200, alpha=0.7, label=label)
    ax.set_xlabel('Avg Samples Used')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Compute Efficiency')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    efficiency_plot_path = comparison_dir / "efficiency_scatter.png"
    plt.savefig(efficiency_plot_path, dpi=150)
    plt.close()
    print(f"Saved: {efficiency_plot_path}")


def create_per_run_plots(run_id: str, run_data: Dict, results_dir: Path):
    """Create plots for individual run."""
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # If we have history data, plot it
    history = run_data.get("wandb_data", {}).get("history", [])
    
    if history:
        # Extract time series
        steps = [h.get("progress", i) for i, h in enumerate(history)]
        running_acc = [h.get("running_accuracy", 0) for h in history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, running_acc, marker='o', linewidth=2)
        ax.set_xlabel('Examples Processed')
        ax.set_ylabel('Running Accuracy')
        ax.set_title(f'Learning Curve: {run_id}')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = run_dir / "learning_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print("="*60)
    print(f"Evaluating {len(run_ids)} runs")
    print(f"Run IDs: {run_ids}")
    print("="*60)
    
    # Get WandB credentials
    wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT")
    
    # Collect data from all runs
    run_data = {}
    
    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        
        # Load local metrics
        local_metrics = load_local_metrics(results_dir, run_id)
        
        # Fetch WandB data if credentials available
        wandb_data = {}
        if wandb_entity and wandb_project:
            wandb_data = fetch_wandb_run(wandb_entity, wandb_project, run_id)
        
        run_data[run_id] = {
            "local_metrics": local_metrics,
            "wandb_data": wandb_data
        }
        
        # Export per-run metrics
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_export_path = run_dir / "metrics.json"
        with open(metrics_export_path, "w") as f:
            json.dump(local_metrics, f, indent=2)
        print(f"Exported metrics: {metrics_export_path}")
        
        # Create per-run plots
        create_per_run_plots(run_id, run_data[run_id], results_dir)
    
    # Create comparison metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate metrics
    metrics_by_run = {}
    for run_id in run_ids:
        metrics = run_data[run_id].get("local_metrics", {})
        if not metrics:
            metrics = run_data[run_id].get("wandb_data", {}).get("summary", {})
        metrics_by_run[run_id] = metrics
    
    # Identify proposed vs baseline
    proposed_runs = [rid for rid in run_ids if "proposed" in rid]
    baseline_runs = [rid for rid in run_ids if "comparative" in rid]
    
    best_proposed_acc = max([metrics_by_run[rid].get("accuracy", 0) for rid in proposed_runs]) if proposed_runs else 0
    best_baseline_acc = max([metrics_by_run[rid].get("accuracy", 0) for rid in baseline_runs]) if baseline_runs else 0
    
    aggregated_metrics = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed_acc,
        "best_baseline": best_baseline_acc,
        "gap": best_proposed_acc - best_baseline_acc
    }
    
    aggregated_path = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nExported aggregated metrics: {aggregated_path}")
    
    # Create comparison plots
    create_comparison_plots(run_data, results_dir)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Best Proposed Accuracy: {best_proposed_acc:.4f}")
    print(f"Best Baseline Accuracy: {best_baseline_acc:.4f}")
    print(f"Gap: {aggregated_metrics['gap']:.4f}")
    print("="*60)
    
    # Print all generated files
    print("\nGenerated files:")
    for run_id in run_ids:
        print(f"  {results_dir / run_id / 'metrics.json'}")
    print(f"  {aggregated_path}")
    print(f"  {comparison_dir / 'accuracy_comparison.png'}")
    print(f"  {comparison_dir / 'avg_samples_comparison.png'}")
    print(f"  {comparison_dir / 'avg_tokens_comparison.png'}")
    print(f"  {comparison_dir / 'efficiency_scatter.png'}")


if __name__ == "__main__":
    main()
