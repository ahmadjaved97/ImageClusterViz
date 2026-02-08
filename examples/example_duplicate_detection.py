"""
Example usage of duplicate detection module.
"""

import os
from imageatlas import DuplicateDetector, DuplicateResults

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

    # Create visualizations (REVISIT)

    print(f"\nFiles to keep: {results.get_best_images()[:5]}")
    print(f"Files to remove: {results.get_images_to_remove()[:5]}")



def main():
    """
    Run all examples
    """

    examples = [
        example_1_simple_hash,
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