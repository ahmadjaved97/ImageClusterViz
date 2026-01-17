import os
import tempfile
import shutil
import numpy as np
from PIL import Image

def create_test_images(output_dir, n_images=20):
    """Create test images for testing."""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_images):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f'test_img_{i:03d}.jpg'))
    
    print(f"✓ Created {n_images} test images in {output_dir}")

def test_new_api():
    """
    Test the new FeaturePipeline API.
    """
    print(f"\n" + "="*60)
    print("TEST 1: New FeaturePipeline API")
    print("="*60)

    try:
        from features import FeaturePipeline
        from feature_extractors import create_feature_extractor

        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        cache_dir = os.path.join(temp_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Create test images
        create_test_images(image_dir, n_images=20)

        # Create extractor
        print("\n1. Creating extractor...")
        extractor = create_feature_extractor('vit', variant='b_16', device='cpu')
        print("   ✓ Extractor created")

        # Create Pipeline
        print("\n2. Creating pipeline...")
        pipeline = FeaturePipeline(
            extractor,
            batch_size=4,
            device='cpu',
            verbose=True
        )
        print("   ✓ Pipeline created")

        # Extract features
        print("\n3. Extracting features...")
        pipeline.extract_from_directory(image_dir)
        print(f"   ✓ Features extracted")

        # Check results
        features = pipeline.get_features()
        filenames = pipeline.get_filenames()
        metadata = pipeline.get_metadata()

        print("\n4.  Validation")
        print(f"   ✓ Features shape: {features.shape}")
        print(f"   ✓ Number of files: {len(filenames)}")
        print(f"   ✓ Feature dimension: {metadata.feature_dim}")

        assert features.shape[0] == 20, "Should have 20 samples"
        assert len(filenames) == 20, "Should have 20 filenames"
        assert features.shape[1] > 0, "Feature dimension should be > 0"

        # Save cache
        cache_path = os.path.join(cache_dir, 'test_features.h5')
        pipeline.save(cache_path)
        assert os.path.exists(cache_path), "Cache file should exist"
        print(f"   ✓ Saved to {cache_path}")

        # Load cache
        print("\n6. Loading from HDF5...")
        pipeline2 = FeaturePipeline(extractor, device='cpu', verbose=False)
        pipeline2.load(cache_path)

        loaded_features = pipeline2.get_features()
        assert np.allclose(features, loaded_features), "Loaded features should match"
        print(f"   ✓ Loaded successfully")

        #Print metadata
        print("\n7. Metadata.")
        print(metadata.summary())

        # Cleanup
        shutil.rmtree(temp_dir)
        print("\n TEST 1 PASSED!")

        return True
    
    except Exception as e:
        print(f"\n TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("FEATURE EXTRACTION PIPELINE TEST SUITE")
    print("="*60)

    results = []

    results.append(("New API", test_new_api()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = " PASSED" if passed else " FAILED"
        print(f"{name:30s} {status}")
    
    all_passed = all(result[1] for result in results)

    print("\n" + "="*60)
    if all_passed:
        print(" ALL TESTS PASSED!")
    else:
        print(" SOME TESTS FAILED!")
        print("="*60)
        print("\nPlease check the error message above.")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
