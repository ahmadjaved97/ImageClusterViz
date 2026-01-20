"""
Complete workflow example.
"""


from core import ImageClusterer

def main():
    """
    Run complete clustering workflow.
    """

    print("="*70)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("="*70)

    #============ STEP 1: CONFIGURE ==============
    print("\nStep 1: Configuration")
    print("-" * 70)

    # Paths
    image_dir = '/home/s63ajave/datasets/temp_ds'
    output_base = './output'

    # Output subdirectories
    grids_dir = f'{output_base}/grids'
    clusters_dir = f'{output_base}/clusters'

    # Configuration
    config = {
        'model': 'dinov2',
        'n_clusters': 10,
        'clustering_method': 'kmeans',
        'reducer': 'pca',
        'n_components': 128,
        'batch_size': 32,
        'device': 'cuda',
    }

    print(f"Input directory: {image_dir}")
    print(f"Output directory: {output_base}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    #============ STEP 2: CLUSTER IMAGES ==============
    print("\nStep 1: Configuration")
    print("-" * 70)

    # Create clusterer
    clusterer = ImageClusterer(**config, verbose=True)

    # Fit to images (with caching for faster re-runs)
    results = clusterer.fit(
        image_dir,
        cache_path=f'{output_base}/features_cache.h5',
        use_cache=False  # Set to True to laod cached feature
    )

    # Print summary
    print("\n" + results.summary())
    

    #============ STEP 3: CREATE VISUALIZATIONS ==============
    print("\nStep 3: Creating Visualizations")
    print("-" * 70)

    # Create grid images for each cluster
    print("\nCreating cluster grids....")
    grid_paths = results.create_grids(
        image_dir=image_dir,
        output_dir=grids_dir,
        image_size=(300, 300)
    )

    print(f"\n   Created {len(grid_paths)} grid images:")
    for cluster_id, path in sorted(grid_paths.items()):
        print(f"   Cluster {cluster_id}: {path}")
    

    #============ STEP 4: ORGANIZE INTO FOLDERS ==============
    print("\nStep 4: Organizing into Folders")
    print("-" * 70)

    # Copy images into cluster folders
    print(f"\nCreating cluster folders....")
    folder_paths = results.create_cluster_folders(
        image_dir=image_dir,
        output_dir=clusters_dir,
        copy_images=True    # True = copy, False = move
    )

    print(f"\n   Created {len(folder_paths)} cluster folders:")
    for cluster_id, path in sorted(folder_paths.items()):
        n_images = len(results.get_cluster(cluster_id))
        print(f"   Cluster {cluster_id}: {path} ({n_images} images)")
    

    # ========== STEP 5: EXPORT RESULTS ==========
    print("\n\nStep 5: Exporting Results")
    print("-" * 70)

    # Export to multiple formats
    print("\nExporting to various formats...")

    # CSV (for spreadsheets)
    results.to_csv(f'{output_base}/clusters.csv')
    print(f"   CSV: {output_base}/clusters.csv")


    # JSON
    results.to_json(f'{output_base}/clusters.json')
    print(f"   JSON: {output_base}/clusters.json")

    # Excel
    try:
        results.to_excel(f'{output_base}/clusters.xlsx')
        print(f"   Excel: {output_base}/clusters.xlsx")
    except ImportError:
        print("   Excel export skipped (install openpyxl)")
    

    # DataFrame (for data analysis)
    df = results.to_dataframe()
    print(f"   DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    # ========== STEP 6: ANALYZE RESULTS ==========
    print("\n\nStep 6: Analysis")
    print("-" * 70)
    
    # Cluster statistics
    print("\nCluster Statistics:")
    sizes = results.get_cluster_sizes()
    for cluster_id in sorted(sizes.keys()):
        size = sizes[cluster_id]
        percentage = (size / len(results.filenames)) * 100
        print(f"  Cluster {cluster_id:2d}: {size:4d} images ({percentage:5.1f}%)")
    
    # Check for outliers (if HDBSCAN was used)
    outliers = results.get_outliers()
    if outliers:
        print(f"\n⚠ Found {len(outliers)} outliers")
    
    
    # ========== DONE ==========
    print("\n\n" + "="*70)
    print("✅ WORKFLOW COMPLETE!")
    print("="*70)
    
    print(f"\nOutputs created:")
    print(f"  1. Grid images:     {grids_dir}/")
    print(f"  2. Cluster folders: {clusters_dir}/")
    print(f"  3. CSV export:      {output_base}/clusters.csv")
    print(f"  4. JSON export:     {output_base}/clusters.json")
    print(f"  5. Feature cache:   {output_base}/features_cache.h5")
    
    print(f"\nYou can now:")
    print(f"  - View cluster grids in {grids_dir}/")
    print(f"  - Browse organized images in {clusters_dir}/")
    print(f"  - Analyze results in {output_base}/clusters.csv")
    print(f"  - Re-run faster using cached features")
    
    return results

    

if __name__ == "__main__":
    results = main()