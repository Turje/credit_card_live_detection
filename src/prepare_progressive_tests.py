"""
Generate progressive occlusion test sets from test split.
Creates test sets with 0%, 25%, 50%, 75% occlusion.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.obscure import PartialOcclusionGenerator


def generate_progressive_tests(
    test_dataset_path: str,
    output_base: str = None,
    occlusion_type: str = "patch",
    seed: int = 42
):
    """
    Generate progressive occlusion test sets.
    
    Args:
        test_dataset_path: Path to test dataset directory
        output_base: Base output directory (default: same as test_dataset_path parent)
        occlusion_type: Type of occlusion
        seed: Random seed
    """
    test_path = Path(test_dataset_path)
    
    if output_base is None:
        output_base = test_path.parent
    
    output_base = Path(output_base)
    
    # Test set with 0% occlusion (original)
    print("Creating baseline test set (0% occlusion)...")
    test_0_path = output_base / "test_occlusion_0"
    test_0_path.mkdir(exist_ok=True)
    (test_0_path / "train").mkdir(exist_ok=True)
    
    # Copy original test set
    import shutil
    import json
    
    # Find source directory (could be test/ or test/train/)
    src_dir = test_path
    if (test_path / "train").exists():
        src_dir = test_path / "train"
    
    # Copy images
    for img_file in src_dir.glob("*.jpg"):
        shutil.copy2(img_file, test_0_path / "train" / img_file.name)
    
    # Copy annotations
    ann_file = src_dir / "_annotations.coco.json"
    if ann_file.exists():
        shutil.copy2(ann_file, test_0_path / "train" / "_annotations.coco.json")
    
    print(f"✅ Baseline test set created: {test_0_path}")
    
    # Generate occluded test sets
    occlusion_levels = [0.25, 0.50, 0.75]
    
    for occlusion_ratio in occlusion_levels:
        print(f"\nGenerating test set with {occlusion_ratio*100:.0f}% occlusion...")
        
        output_path = output_base / f"test_occlusion_{int(occlusion_ratio*100)}"
        
        # Use test_path directly (obscure.py expects train/ subdirectory)
        # If test_path is already the test directory, we need to create a temp structure
        temp_test = test_path
        if not (test_path / "train").exists():
            # Create a temporary structure for obscure.py
            temp_test = test_path.parent / f"_temp_{test_path.name}"
            temp_test.mkdir(exist_ok=True)
            (temp_test / "train").mkdir(exist_ok=True)
            # Copy files
            for img_file in test_path.glob("*.jpg"):
                shutil.copy2(img_file, temp_test / "train" / img_file.name)
            ann_file = test_path / "_annotations.coco.json"
            if ann_file.exists():
                shutil.copy2(ann_file, temp_test / "train" / "_annotations.coco.json")
        
        generator = PartialOcclusionGenerator(str(temp_test))
        generator.generate_obscured_dataset(
            output_path=str(output_path),
            occlusion_type=occlusion_type,
            occlusion_ratio=occlusion_ratio,
            random_seed=seed
        )
        
        # Clean up temp directory if created
        if temp_test != test_path and temp_test.exists():
            shutil.rmtree(temp_test)
        
        print(f"✅ Test set created: {output_path}")
    
    print(f"\n✅ All progressive test sets created!")
    print(f"Test sets available:")
    print(f"  - test_occlusion_0 (baseline)")
    for ratio in occlusion_levels:
        print(f"  - test_occlusion_{int(ratio*100)} ({ratio*100:.0f}% occlusion)")


def main():
    parser = argparse.ArgumentParser(description="Generate progressive occlusion test sets")
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Path to test dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base directory (default: same as test dataset parent)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="crop",
        choices=["patch", "blur", "noise", "black", "white", "crop", "random"],
        help="Occlusion type: 'crop' (camera pan/zoom, recommended), 'patch' (random patches), etc. (default: crop)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_progressive_tests(
            test_dataset_path=args.test_dataset,
            output_base=args.output,
            occlusion_type=args.type,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

