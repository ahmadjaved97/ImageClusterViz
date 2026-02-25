"""
Example usage of duplicate detection module.
"""

import os
from imageatlas import DuplicateDetector, DuplicateResults, create_duplicate_grids

def example_1_simple_hash():
    """
    Example 1: Simple perceptual hash detection.
    """

    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Perceptual Hash Detection")
    print("="*70)

    # Create detector with perceptual hash
    detector = DuplicateDetector(
        method='phash',
        threshold=0.8,
        grouping=True,
        best_selection='resolution'
    )

    # Detect duplicates
    results = detector.detect("/home/s63ajave/datasets/temp_ds")

    # Print summary
    print(results.summary())

    # Export results
    results.to_csv("./output/duplicates.csv")
    results.to_json("./output/duplicates.json")

    # Create visualizations
    create_duplicate_grids(
        results,
        image_dir="/home/s63ajave/datasets/temp_ds",
        output_dir='./output/grids',
        top_n=10
    )

def example_2_deep_learning():
    """
    Deep learning based duplicate detection.
    """

    print("\n" + "="*70)
    print("EXAMPLE 2: Deep Learning Embedding Detection (DINOv2)")
    print("="*70)

    # Create detector with DINOv2 embeddings
    detector = DuplicateDetector(
        method='embedding',
        model='dinov2',
        variant='vits14',
        similarity_metric='cosine',
        threshold=0.95,
        device='cuda',
        batch_size=32,
        use_cache=True,
        cache_path='./cache/features.h5',
        verbose=True,
    )

    #Detect duplicates
    results = detector.detect('/home/s63ajave/datasets/temp_ds')

    # Get statistics
    stats = results.get_statistics()
    print(f"\nDataset reduction: {stats['reduction_percentage']:.1f}%")
    print(f"Can remove {stats['total_duplicates']} images")
    
    # Export to Excel
    results.to_json('./output/duplicates_dinov2.json')

    # Create visualizations
    create_duplicate_grids(
        results,
        image_dir="/home/s63ajave/datasets/temp_ds",
        output_dir='./output/grids_dinov2',
        top_n=10
    )






def main():
    """
    Run all examples
    """

    examples = [
        example_1_simple_hash,
        example_2_deep_learning
    ]

    print("\n" + "="*70)
    print("IMAGEATLAS DUPLICATE DETECTION EXAMPLES")
    print("="*70)
    
    for i, example in enumerate(examples, 1):
        print(f"\nRunning example {i}/{len(examples)}...")
        try:
            example()
        except Exception as e:
            print(f"Example {i} failed: {e}")
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()