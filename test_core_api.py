"""
Test script for the core API.
"""

import os
import tempfile
import shutil
import numpy as np
from PIL import Image

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



def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("CORE API TEST SUITE")
    print("="*60)
    
    results = []
    
    results.append(("Basic Clustering", test_basic_clustering()))
    results.append(("CSV Export", test_export_csv()))


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
        print("\nThe core API is working correctly.")
        print("You can now use ImageClusterer in your projects!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("="*60)
        print("\nPlease check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)