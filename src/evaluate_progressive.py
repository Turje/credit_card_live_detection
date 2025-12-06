"""
Evaluate model on progressive occlusion test sets.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model_on_test_set(model_path: str, test_dataset_path: str):
    """
    Evaluate model on a test set.
    
    Args:
        model_path: Path to trained model
        test_dataset_path: Path to test dataset
    
    Returns:
        Dictionary with metrics
    """
    model = YOLO(model_path)
    test_path = Path(test_dataset_path)
    
    # Find images directory
    images_dir = test_path / "train" / "images"
    if not images_dir.exists():
        images_dir = test_path / "train"
    
    # Run validation
    results = model.val(
        data=str(test_path / "train" / "_annotations.coco.json"),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save_json=True,
        project="outputs/evaluation",
        name=test_path.name
    )
    
    metrics = {
        'mAP50': results.box.map50,
        'mAP50_95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
    }
    
    return metrics


def evaluate_progressive(
    model_path: str,
    test_sets_base: str,
    output_dir: str = "outputs/progressive_evaluation"
):
    """
    Evaluate model on progressive occlusion test sets.
    
    Args:
        model_path: Path to trained model
        test_sets_base: Base directory containing test_occlusion_* folders
        output_dir: Output directory for results
    """
    test_base = Path(test_sets_base)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all test sets
    occlusion_levels = [0, 25, 50, 75]
    results = {}
    
    print("Evaluating model on progressive occlusion test sets...\n")
    
    for level in occlusion_levels:
        test_set_path = test_base / f"test_occlusion_{level}"
        
        if not test_set_path.exists():
            print(f"Warning: Test set not found: {test_set_path}")
            continue
        
        print(f"Evaluating on {level}% occlusion...")
        
        try:
            metrics = evaluate_model_on_test_set(model_path, str(test_set_path))
            results[level] = metrics
            
            print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}\n")
        
        except Exception as e:
            print(f"  Error evaluating {level}% occlusion: {e}\n")
            continue
    
    # Save results to JSON
    results_file = output_dir / "progressive_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Create visualization
    plot_progressive_results(results, output_dir)
    
    return results


def plot_progressive_results(results: dict, output_dir: Path):
    """Plot progressive occlusion results."""
    if not results:
        print("No results to plot")
        return
    
    occlusion_levels = sorted(results.keys())
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[level][metric] for level in occlusion_levels]
        
        ax.plot(occlusion_levels, values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Occlusion Level (%)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Occlusion Level')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for level, value in zip(occlusion_levels, values):
            ax.text(level, value + 0.02, f'{value:.3f}', ha='center', fontsize=9)
    
    # Combined plot
    ax = axes[5]
    for metric in ['mAP50', 'precision', 'recall']:
        values = [results[level][metric] for level in occlusion_levels]
        ax.plot(occlusion_levels, values, marker='o', label=metric.upper(), linewidth=2)
    
    ax.set_xlabel('Occlusion Level (%)')
    ax.set_ylabel('Score')
    ax.set_title('Key Metrics vs Occlusion Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plot_file = output_dir / "progressive_occlusion_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_file}")
    
    # Create summary table
    df = pd.DataFrame(results).T
    df.index.name = 'Occlusion_Level'
    csv_file = output_dir / "progressive_results.csv"
    df.to_csv(csv_file)
    print(f"CSV saved to: {csv_file}")
    
    print("\n" + "="*60)
    print("PROGRESSIVE OCCLUSION EVALUATION SUMMARY")
    print("="*60)
    print(df.to_string())
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on progressive occlusion test sets")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--test-sets",
        type=str,
        required=True,
        help="Base directory containing test_occlusion_* folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/progressive_evaluation",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_progressive(
            model_path=args.model,
            test_sets_base=args.test_sets,
            output_dir=args.output
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

