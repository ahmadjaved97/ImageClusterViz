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


def example_3_adaptive_threshold():
    """Example 3: Adaptive threshold selection."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Adaptive Threshold Selection")
    print("="*70)

    detector = DuplicateDetector(
        method='phash',
        threshold=None,
        adaptive_percentile=95,
        grouping=True,
        verbose=True
    )

    # Detect duplicates
    results = detector.detect("./images")

    # Check what threshold was selected
    print(f"\nAuto-selected threshold: {results.metadata['threshold']:.3f}")


def example_4_clip_detection():
    """
    Example 4: CLIP-based semantic similarity.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CLIP Semantic Similarity")
    print("="*70)

    # Create detector with CLIP
    detector = DuplicateDetector(
        method='clip',
        variant='ViT-B/16',
        threshold=0.92,
        device='cuda',
        batch_size=16
    )

    # Detect duplicates
    results = detector.detect("./images")

    print(results.summary())

def example_5_from_embeddings():
    """
    Example 5: Detect from pre-computed embeddings.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Detection from Pre-computed Embeddings")
    print("="*70)

    # First, extract features (maybe already done for clustering)
    from imageatlas import FeaturePipeline, create_feature_extractor

     extractor = create_feature_extractor('dinov2', variant='vits14')
    pipeline = FeaturePipeline(extractor, batch_size=32)
    pipeline.extract_from_directory('./images')
    
    embeddings = pipeline.get_features()
    filenames = pipeline.get_filenames()
    
    # Now detect duplicates from these embeddings
    detector = DuplicateDetector(
        method='embedding',  # Still need to specify method
        threshold=0.9
    )
    
    results = detector.detect_from_embeddings(embeddings, filenames)
    print(results.summary())


def example_6_cluster_then_deduplicate():
    """Example 6: Cluster first, then find duplicates within clusters."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Cluster then Deduplicate")
    print("="*70)
    
    # First, cluster images
    from imageatlas import ImageClusterer
    
    clusterer = ImageClusterer(
        model='dinov2',
        n_clusters=5,
        clustering_method='kmeans'
    )
    
    cluster_results = clusterer.fit('./images')
    
    # Now find duplicates within each cluster
    detector = DuplicateDetector(
        method='phash',
        threshold=0.9
    )
    
    for cluster_id in range(cluster_results.n_clusters):
        print(f"\n--- Cluster {cluster_id} ---")
        
        # Get images in this cluster
        cluster_images = cluster_results.get_cluster(cluster_id)
        
        # Find duplicates within cluster
        dup_results = detector.detect(cluster_images)
        
        print(f"Found {len(dup_results.groups)} duplicate groups")
        
        # Save cluster-specific results
        dup_results.to_csv(f'./output/cluster_{cluster_id}_duplicates.csv')



def example_7_load_cached_results():
    """Example 7: Load previously saved results."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Load Cached Results")
    print("="*70)
    
    # Detect and save
    detector = DuplicateDetector(method='phash', threshold=0.9)
    results = detector.detect('./images')
    results.save('./output/duplicates.h5')
    
    # Later, load without recomputing
    loaded_results = DuplicateResults.load('./output/duplicates.h5')
    
    print(loaded_results.summary())
    
    # Re-export in different format
    loaded_results.to_csv('./output/duplicates_reloaded.csv')



def example_8_custom_selection():
    """Example 8: Custom best image selection criteria."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Custom Selection Criteria")
    print("="*70)
    
    # Use file size instead of resolution
    detector = DuplicateDetector(
        method='phash',
        threshold=0.9,
        best_selection='filesize'  # Keep largest files
    )
    
    results = detector.detect('./images')
    print(results.summary())


def example_10_comprehensive_workflow():
    """Example 10: Complete workflow with all features."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Comprehensive Workflow")
    print("="*70)
    
    # Configuration
    image_dir = './images'
    output_dir = './output'
    
    # Step 1: Detect duplicates
    detector = DuplicateDetector(
        method='embedding',
        model='dinov2',
        variant='vits14',
        threshold=0.95,
        grouping=True,
        best_selection='both',  # Resolution + alphabetic
        device='cuda',
        batch_size=32,
        use_cache=True,
        cache_path=f'{output_dir}/cache.h5',
        verbose=True
    )
    
    results = detector.detect(image_dir)
    
    # Step 2: Print comprehensive statistics
    print("\n" + results.summary())
    
    stats = results.get_statistics()
    print(f"\nDetailed Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Step 3: Export to all formats
    results.to_csv(f'{output_dir}/duplicates.csv')
    results.to_json(f'{output_dir}/duplicates.json')
    results.to_excel(f'{output_dir}/duplicates.xlsx')
    
    # Step 4: Create visualizations
    grid_paths = create_duplicate_grids(
        results,
        image_dir=image_dir,
        output_dir=f'{output_dir}/grids',
        image_size=(300, 300),
        top_n=10
    )
    
    print(f"\nCreated {len(grid_paths)} grid visualizations")
    
    # Step 5: Save results for later
    results.save(f'{output_dir}/duplicates.h5')
    
    # Step 6: Generate removal script
    to_remove = results.get_images_to_remove()
    
    with open(f'{output_dir}/remove_duplicates.txt', 'w') as f:
        for img in to_remove:
            f.write(f"{img}\n")

    
    print("\n✓ Workflow complete!")

def main():
    """
    Run all examples
    """

    examples = [
        example_1_simple_hash,
        example_2_deep_learning,
        example_3_adaptive_threshold,
        example_4_clip_detection,
        example_5_from_embeddings,
        example_6_cluster_then_deduplicate,
        example_7_load_cached_results,
        example_8_custom_selection,
        example_10_comprehensive_workflow

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