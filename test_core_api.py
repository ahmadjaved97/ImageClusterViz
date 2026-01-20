"""
Test script for the core API.
"""

import os
import tempfile
import shutil
import numpy as np
from PIL import Image
from rich import print

def create_test_images(output_dir, n_images=50):
    """
    Create test images for testing.
    """

    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_images):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f'test_img_{i:03d}.jpg'))
    
    print(f"✓ Created {n_images} test images in {output_dir}")


def test_basic_clustering():
    """
    Test basic clustering workflow.
    """
    print("\n" + "="*60)
    print("TEST 1: Basic Clustering")
    print("="*60)

    try:
        from core import ImageClusterer

        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')

        # Create test images
        create_test_images(image_dir, n_images=30)

        # Create clusterer
        print(f"\n1. Creating ImageClusterer...")
        clusterer = ImageClusterer(
            model='vit',
            model_variant='b_16',
            n_clusters=3,
            batch_size=4,
            device='cpu',
            verbose=True
        )

        print(f"   Created: {clusterer}")

        # Fit
        print(f"\n2. Fitting to images...")
        results = clusterer.fit(image_dir)
        print(f"   Fitted successfully")
        print(f"   Found {results.n_clusters} clusters")
        print(f"   Cluster sizes: {results.get_cluster_sizes()}")

        # Check results
        assert results.n_clusters == 3
        assert len(results.filenames) == 30
        assert results.features is not None


        # Cleanup
        shutil.rmtree(temp_dir)

        print("\n TEST 1 PASSED")
        return True

    except Exception as e:
        print(f"\n TEST 1 FAILED")
        import traceback
        traceback.print_exc()
        return False


def test_export_csv():
    """Test CSV export."""
    print("\n" + "="*60)
    print("TEST 2: CSV Export")
    print("="*60)
    
    try:
        from core import ImageClusterer
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test images
        create_test_images(image_dir, n_images=20)
        
        # Cluster
        print("\n1. Clustering...")
        clusterer = ImageClusterer(model='vit', n_clusters=3, batch_size=4, verbose=False)
        results = clusterer.fit(image_dir)
        
        # Export CSV
        print("\n2. Exporting to CSV...")
        csv_path = os.path.join(output_dir, 'clusters.csv')
        results.to_csv(csv_path)
        
        # Verify CSV exists
        assert os.path.exists(csv_path)
        print(f"   ✓ CSV created: {csv_path}")
        
        # Read and check CSV
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 20
            assert 'filename' in rows[0]
            assert 'cluster_id' in rows[0]
        print(f"   CSV has correct structure")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_reduction():
    """Test clustering with dimensionality reduction."""
    print("\n" + "="*60)
    print("TEST 4: With Dimensionality Reduction")
    print("="*60)
    
    try:
        from core import ImageClusterer
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        
        # Create test images
        create_test_images(image_dir, n_images=35)
        
        # Cluster with PCA
        print("\n1. Clustering with PCA...")
        clusterer = ImageClusterer(
            model='vit',
            reducer='pca',
            n_components=32,
            n_clusters=3,
            batch_size=4,
            verbose=False
        )
        results = clusterer.fit(image_dir)
        
        print(f"    Clustered with PCA")
        assert results.reduced_features is not None
        assert results.reduced_features.shape[1] == 32
        print(f"    Reduced to {results.reduced_features.shape[1]} dimensions")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fit_features():
    """Test fitting with pre-computed features."""
    print("\n" + "="*60)
    print("TEST 5: Fit Pre-computed Features")
    print("="*60)
    
    try:
        from core import ImageClusterer
        
        # Create fake features
        print("\n1. Creating fake features...")
        features = np.random.randn(50, 128)
        filenames = [f'img_{i:03d}.jpg' for i in range(50)]
        
        # Cluster
        print("\n2. Clustering features...")
        clusterer = ImageClusterer(
            clustering_method='kmeans',
            n_clusters=5,
            verbose=False
        )
        results = clusterer.fit_features(features, filenames)
        
        print(f"    Clustered {len(filenames)} samples")
        print(f"    Found {results.n_clusters} clusters")
        assert results.n_clusters == 5
        assert len(results.filenames) == 50
        
        print("\n TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataframe_export():
    """Test DataFrame export."""
    print("\n" + "="*60)
    print("TEST 6: DataFrame Export")
    print("="*60)
    
    try:
        from core import ImageClusterer
        
        # Create fake features
        features = np.random.randn(30, 64)
        filenames = [f'img_{i:03d}.jpg' for i in range(30)]
        
        # Cluster
        print("\n1. Clustering...")
        clusterer = ImageClusterer(n_clusters=3, verbose=False)
        results = clusterer.fit_features(features, filenames)
        
        # Export to DataFrame
        print("\n2. Converting to DataFrame...")
        df = results.to_dataframe()
        
        print(f"   ✓ DataFrame shape: {df.shape}")
        assert df.shape[0] == 30
        assert 'filename' in df.columns
        assert 'cluster_id' in df.columns
        print(f"   ✓ Has correct columns: {list(df.columns)}")
        
        print("\n TEST 6 PASSED")
        return True
        
    except ImportError:
        print("\n  TEST 6 SKIPPED: pandas not installed")
        return True
    except Exception as e:
        print(f"\n TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_export_json():
    """Test JSON export."""
    print("\n" + "="*60)
    print("TEST 3: JSON Export")
    print("="*60)
    
    try:
        from core import ImageClusterer
        import json
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test images
        create_test_images(image_dir, n_images=20)
        
        # Cluster
        print("\n1. Clustering...")
        clusterer = ImageClusterer(model='vit', n_clusters=3, batch_size=4, verbose=False)
        results = clusterer.fit(image_dir)
        
        # Export JSON
        print("\n2. Exporting to JSON...")
        json_path = os.path.join(output_dir, 'clusters.json')
        results.to_json(json_path)
        
        # Verify JSON exists
        assert os.path.exists(json_path)
        print(f"    JSON created: {json_path}")
        
        # Read and check JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert 'n_samples' in data
            assert 'n_clusters' in data
            assert 'clusters' in data
            assert data['n_samples'] == 20
            assert data['n_clusters'] == 3
        print(f"    JSON has correct structure")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("CORE API TEST SUITE")
    print("="*60)
    
    results = []
    
    results.append(("Basic Clustering", test_basic_clustering()))
    results.append(("CSV Export", test_export_csv()))
    results.append(("With Reduction", test_with_reduction()))
    results.append(("Fit Features", test_fit_features()))
    results.append(("DataFrame Export", test_dataframe_export()))
    results.append(("JSON Export", test_export_json()))


    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
    else:
        print("❌ SOME TESTS FAILED!")
        print("="*60)
        print("\nPlease check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)