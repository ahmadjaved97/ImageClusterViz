"""
Test script for reduction.
"""

import numpy as np
from reduction import (
    create_reducer,
    get_available_reducers,
    PCAReducer,
    UMAPReducer,
    TSNEReducer
)


def test_factory():
    """
    Test factory function.
    """
    print("\n" + "="*60)
    print("TEST 1: Factory Function")
    print("\n" + "="*60)

    try:
        # Get available algorithms
        algorithms = get_available_reducers()
        print(f"\n  Available algorithms: {algorithms}")

        # Create each algorithm
        for algo in algorithms:
            reducer = create_reducer(algo, n_components=10)
            print(f"   Created {algo}: {reducer}")
        
        print("\n TEST 1 PASSED")
        return True
    except Exception as e:
        print(f"\n TEST 1 FAILED: {e}")
        return False

def test_pca():
    """
    Test PCA Reducer.
    """
    print("\n" + "="*60)
    print("\n2. PCA Reducer")
    print("="*60)

    try:
        # Create sample data
        features = np.random.randn(100, 50)

        # Test standard PCA
        print("\n1. Standard PCA")
        pca = PCAReducer(n_components=10)
        reduced = pca.fit_transform(features)

        print(f"   Input shape: {features.shape}")
        print(f"   Output shape: {reduced.shape}")
        assert reduced.shape == (100, 10)

        # Test variance explained
        variance = pca.get_explained_variance()
        cumulative = pca.get_cumulative_variance()
        print(f"   Total variance explained: {sum(variance):.2%}")
        print(f"   Cumulative variance shape: {cumulative.shape}")

        # Test metadata
        metadata = pca.get_metadata()
        print(f"   Metadata keys: {list(metadata.keys())}")

        # TEst transform on new data
        new_features = np.random.randn(20, 50)
        new_reduced = pca.transform(new_features)
        assert new_reduced.shape == (20, 10)
        print(f"   Transform new data: {new_reduced.shape}")


        # Test incremental PCA
        print("\n2. Incremental PCA:")
        pca_inc = PCAReducer(n_components=10, use_incremental=True)
        reducer_inc = pca_inc.fit_transform(features)
        assert reducer_inc.shape == (100, 10)
        print(f"   Incremental PCA works: {reducer_inc.shape}")

        print("\n  TEST 2 PASSED")
        return True
    except Exception as e:
        print(f"\n   TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_umap():
    """
    Test UMAP Reducer.
    """
    print("\n" + "="*60)
    print("\n2. UMAP Reducer")
    print("="*60)


    try:
        # Create sample data
        features = np.random.randn(100, 50)

        # Test standard PCA
        print("\n1. UMAP Reduction")
        umap = UMAPReducer(n_components=10)
        reduced = umap.fit_transform(features)

        print(f"   Input shape: {features.shape}")
        print(f"   Output shape: {reduced.shape}")
        assert reduced.shape == (100, 10)

        # Test metadata
        metadata = umap.get_metadata()
        print(f"   Metadata keys: {list(metadata.keys())}")

        # TEst transform on new data
        new_features = np.random.randn(20, 50)
        new_reduced = umap.transform(new_features)
        assert new_reduced.shape == (20, 10)
        print(f"   Transform works: {new_reduced.shape}")

        # Test 2D for visualization
        print("\n3. 2D visualization:")
        umap_2d = UMAPReducer(n_components=2)
        embedding = umap_2d.fit_transform(features)
        assert embedding.shape == (100, 2)
        print(f"   ✓ 2D embedding: {embedding.shape}")
        
        print("\n TEST 3 PASSED")
        return True
    except ImportError as e:
        print(f"\n TEST 3 SKIPPED: UMAP not installed")
        print(f"   Install with: pip install umap-learn")
        return True  # Don't fail if UMAP not installed
    except Exception as e:
        print(f"\n TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tsne():
    """Test t-SNE reducer."""
    print("\n" + "="*60)
    print("TEST 4: t-SNE Reducer")
    print("="*60)
    
    try:
        # Create sample data
        features = np.random.randn(100, 50)
        
        print("\n1. t-SNE fit_transform:")
        tsne = TSNEReducer(n_components=2, perplexity=30)
        embedding = tsne.fit_transform(features)
        
        print(f"   ✓ Input shape: {features.shape}")
        print(f"   ✓ Output shape: {embedding.shape}")
        assert embedding.shape == (100, 2)
        
        # Test metadata
        metadata = tsne.get_metadata()
        print(f"   ✓ Metadata keys: {list(metadata.keys())}")
        if 'kl_divergence' in metadata:
            print(f"   ✓ KL divergence: {metadata['kl_divergence']:.4f}")
        
        # Test that transform is not supported
        print("\n2. Test transform() raises error:")
        try:
            new_features = np.random.randn(20, 50)
            tsne.transform(new_features)
            print("   Should have raised NotImplementedError")
            return False
        except NotImplementedError:
            print("   ✓ Correctly raises NotImplementedError")
        
        print("\nTEST 4 PASSED")
        return True
    
    except Exception as e:
        print(f"\nTEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test integration with clustering pipeline."""
    print("\n" + "="*60)
    print("TEST 5: Pipeline Integration")
    print("="*60)
    
    try:
        # Simulate feature extraction output
        features = np.random.randn(200, 128)
        filenames = [f"img_{i:03d}.jpg" for i in range(200)]
        
        print("\n1. Reduce dimensions:")
        reducer = create_reducer('pca', n_components=32)
        reduced = reducer.fit_transform(features)
        print(f"   ✓ Reduced from {features.shape[1]}D to {reduced.shape[1]}D")
        
        print("\n2. Test with clustering:")
        from clustering.factory import create_clustering_algorithm
        
        clusterer = create_clustering_algorithm('kmeans', n_clusters=5)
        result = clusterer.fit_predict(reduced, filenames=filenames)
        
        print(f"   ✓ Clustered into {result.n_clusters} clusters")
        print(f"   ✓ Cluster sizes: {result.get_cluster_sizes()}")
        
        print("\n TEST 5 PASSED")
        return True
    except Exception as e:
        print(f"\n TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def run_all_tests():
    """
    Run all tests.
    """
    print("="*60)
    print("REDUCTION MODULE TEST SUITE")
    print("="*60)
    
    results = []

    results.append(("Factory Function", test_factory()))
    results.append(("PCA Reducer", test_pca()))
    results.append(("UMAP Reducer", test_umap()))
    results.append(("t-SNE Reducer", test_tsne()))
    results.append(("Pipeline Integration", test_pipeline_integration()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = " PASSED" if passed else "FAILED"
        print(f"{name:30s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nThe reduction module is working correctly.")
    else:
        print("SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the error messages above.")
    
    return all_passed

if __name__ =="__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)