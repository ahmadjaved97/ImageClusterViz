import time
import numpy as np
from PIL import Image
from feature_extractors import create_feature_extractor


def create_dummy_images(n_images=100, size=(224, 224)):
    """Create dummy images for testing."""
    images = []
    for i in range(n_images):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images


def benchmark_one_by_one(extractor, images):
    """
    One by one extraction.
    """
    print(f"\n1. One-by-one extraction.")

    start = time.time()
    features_list = []

    for img in images:
        feature = extractor.extract_features(img)
        features_list.append(feature)
    
    features = np.vstack(features_list)
    elapsed = time.time() - start

    speed = (len(images)) / elapsed
    print(f"   Time: {elapsed:.3f}")
    print(f"   Speed: {speed:.1f} images/sec")
    print(f"   Shape: {features.shape}")

    return elapsed, speed


def benchmark_batch(extractor, images, batch_size=32):
    print(f"\n2. Batch extraction. Batch size = {batch_size}")

    if not hasattr(extractor, 'extract_batch'):
        print(f"   Extractor doesn't support extract batch.")
        return None, None
    
    start = time.time()

    # Process in batches
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i: i+batch_size]
        batch_features = extractor.extract_batch(batch)
        all_features.append(batch_features)
    
    features = np.vstack(all_features)
    elapsed = time.time() - start

    speed = len(images) / elapsed
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Speed: {speed:.1f} images/sec")
    print(f"   Shape: {features.shape}")

    return elapsed, speed


def main():
    print("="*60)
    print("BATCH PROCESSING BENCHMARK")
    print("="*60)

    # Configuration
    n_images = 2000
    batch_sizes = [4, 8, 16, 32, 64, 128]

    print(f"\nTest Configuration")
    print(f"   Number of images: {n_images}")
    print(f"   Image size: 224x224")
    print(f"   Model: ViT/B-16")
    print(f"   Device: CPU")

    # Create test data
    print(f"\nCreating {n_images} dummy images....")
    images = create_dummy_images(n_images)
    print(f"  Images created")

    # Create extractor
    extractor = create_feature_extractor('vit', variant='b_16', device='cuda')
    print("   Extractor created")

    # Benchmark one-by-one
    time_single, speed_single = benchmark_one_by_one(extractor, images)

    # Benchmark different batch sizes
    results = []
    for batch_size in batch_sizes:
        time_batch, speed_batch = benchmark_batch(extractor, images, batch_size)
        if time_batch is not None:
            speedup = speed_batch / speed_single
            results.append((batch_size, time_batch, speed_batch, speedup))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nOne-by-one (baseline):")
    print(f"  Speed: {speed_single:.1f} images/sec")
    
    print(f"\nBatch processing:")
    for batch_size, time_batch, speed_batch, speedup in results:
        print(f"  Batch size {batch_size:2d}: {speed_batch:6.1f} images/sec  "
              f"({speedup:.2f}x speedup)")
    
    # Best result
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"\n✨ Best performance: batch_size={best[0]} "
              f"({best[3]:.2f}x faster than one-by-one)")
    
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE")
    print("="*60)
    print("\nNote: Speedup will be even greater on GPU!")


if __name__ == "__main__":
    main()