"""
Test script for visualization functionality.

Run this to verify grid creation is working:
    python test_visualization.py
"""

import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import cv2

def create_test_images(output_dir, n_images=30, create_variety=True):
    """Create test images with some visual variety."""
    os.makedirs(output_dir, exist_ok=True)
    
    if create_variety:
        # Create images with different colors for visual distinction
        colors = [
            (255, 100, 100),  # Red-ish
            (100, 255, 100),  # Green-ish
            (100, 100, 255),  # Blue-ish
        ]
        
        for i in range(n_images):
            color = colors[i % len(colors)]
            # Create colored image with some noise
            img_array = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img_array = img_array + np.array(color)
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            img.save(os.path.join(output_dir, f'test_img_{i:03d}.jpg'))
    else:
        # Create random images
        for i in range(n_images):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(output_dir, f'test_img_{i:03d}.jpg'))
    
    print(f"✓ Created {n_images} test images")


def test_grid_visualizer():
    """Test GridVisualizer class."""
    print("\n" + "="*60)
    print("TEST 1: GridVisualizer")
    print("="*60)
    
    try:
        from visualization import GridVisualizer
        
        # Create some test images
        print("\n1. Creating test images...")
        images = []
        for i in range(12):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i % 3]
            img = np.full((100, 100, 3), color, dtype=np.uint8)
            images.append(img)
        print(f"   ✓ Created {len(images)} test images")
        
        # Create visualizer
        print("\n2. Creating grid...")
        visualizer = GridVisualizer(image_size=(50, 50))
        grid = visualizer.create_grid(images)
        
        print(f"   ✓ Grid shape: {grid.shape}")
        assert grid.shape[0] > 0 and grid.shape[1] > 0
        print(f"   ✓ Grid created successfully")
        
        print("\n TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_create_cluster_grids():
    """Test create_cluster_grids function."""
    print("\n" + "="*60)
    print("TEST 2: Create Cluster Grids")
    print("="*60)
    
    try:
        from visualization import create_cluster_grids
        
        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        output_dir = os.path.join(temp_dir, 'grids')
        
        # Create test images
        print("\n1. Creating test images...")
        create_test_images(image_dir, n_images=20, create_variety=True)
        
        # Create fake cluster dict
        cluster_dict = {
            0: [f'test_img_{i:03d}.jpg' for i in range(0, 7)],
            1: [f'test_img_{i:03d}.jpg' for i in range(7, 14)],
            2: [f'test_img_{i:03d}.jpg' for i in range(14, 20)],
        }
        
        # Create grids
        print("\n2. Creating cluster grids...")
        grid_paths = create_cluster_grids(
            cluster_dict=cluster_dict,
            image_dir=image_dir,
            output_dir=output_dir,
            image_size=(100, 100),
            verbose=False
        )
        
        print(f"   ✓ Created {len(grid_paths)} grids")
        
        # Verify grids exist
        for cluster_id, path in grid_paths.items():
            assert os.path.exists(path), f"Grid {path} does not exist"
            
            # Check grid is valid image
            img = cv2.imread(path)
            assert img is not None, f"Failed to read grid {path}"
            print(f"   ✓ cluster_{cluster_id}.jpg: {img.shape}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_results_create_grids():
    """Test results.create_grids() method."""
    print("\n" + "="*60)
    print("TEST 3: ClusteringResults.create_grids()")
    print("="*60)
    
    try:
        from core import ImageClusterer
        
        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        grid_dir = os.path.join(temp_dir, 'grids')
        
        # Create test images
        print("\n1. Creating test images...")
        create_test_images(image_dir, n_images=25, create_variety=True)
        
        # Cluster
        print("\n2. Clustering images...")
        clusterer = ImageClusterer(
            model='vit',
            model_variant='b_16',
            n_clusters=3,
            batch_size=4,
            device='cpu',
            verbose=False
        )
        results = clusterer.fit(image_dir)
        print(f"   ✓ Clustered into {results.n_clusters} clusters")
        
        # Create grids
        print("\n3. Creating grids...")
        grid_paths = results.create_grids(
            image_dir=image_dir,
            output_dir=grid_dir,
            image_size=(120, 120),
            verbose=False
        )
        
        print(f"   ✓ Created {len(grid_paths)} grid images")
        
        # Verify
        for cluster_id, path in grid_paths.items():
            assert os.path.exists(path)
            print(f"   ✓ {os.path.basename(path)} exists")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n✅ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_cluster_folders():
    """Test results.create_cluster_folders() method."""
    print("\n" + "="*60)
    print("TEST 4: ClusteringResults.create_cluster_folders()")
    print("="*60)
    
    try:
        from core import ImageClusterer
        
        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        cluster_dir = os.path.join(temp_dir, 'clusters')
        
        # Create test images
        print("\n1. Creating test images...")
        create_test_images(image_dir, n_images=20, create_variety=False)
        
        # Cluster
        print("\n2. Clustering images...")
        clusterer = ImageClusterer(
            model='vit',
            n_clusters=3,
            batch_size=4,
            verbose=False
        )
        results = clusterer.fit(image_dir)


        # After clustering, before creating folders:
        print(f"Image dir: {image_dir}")
        print(f"Files in dir: {os.listdir(image_dir)}")
        print(f"Cluster dict sample: {list(results.cluster_dict.values())[0][:3]}")
        
        # Create cluster folders
        print("\n3. Creating cluster folders...")
        folder_paths = results.create_cluster_folders(
            image_dir=image_dir,
            output_dir=cluster_dir,
            copy_images=True,
            verbose=True
        )
        
        print(f"   ✓ Created {len(folder_paths)} cluster folders")
        
        # Verify folders and images
        for cluster_id, folder_path in folder_paths.items():
            assert os.path.exists(folder_path)
            files = os.listdir(folder_path)
            print(f"   ✓ cluster_{cluster_id}: {len(files)} images")
            assert len(files) > 0
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n✅ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_workflow():
    """Test complete workflow: cluster → grids → folders."""
    print("\n" + "="*60)
    print("TEST 5: Complete Workflow")
    print("="*60)
    
    try:
        from core import ImageClusterer
        
        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        image_dir = os.path.join(temp_dir, 'images')
        grid_dir = os.path.join(temp_dir, 'grids')
        cluster_dir = os.path.join(temp_dir, 'clusters')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test images
        print("\n1. Creating test images...")
        create_test_images(image_dir, n_images=30, create_variety=True)
        
        # Cluster
        print("\n2. Clustering...")
        clusterer = ImageClusterer(
            model='vit',
            n_clusters=4,
            batch_size=8,
            verbose=True
        )
        results = clusterer.fit(image_dir)
        print(f"   ✓ {results.n_clusters} clusters, {len(results.filenames)} images")
        
        # Create grids
        print("\n3. Creating grids...")
        grid_paths = results.create_grids(image_dir, grid_dir, verbose=False)
        print(f"   ✓ {len(grid_paths)} grids created")
        
        # Create cluster folders
        print("\n4. Creating cluster folders...")
        folder_paths = results.create_cluster_folders(
            image_dir, cluster_dir, verbose=True
        )
        print(f"   ✓ {len(folder_paths)} folders created")
        
        # Export results
        print("\n5. Exporting results...")
        results.to_csv(os.path.join(output_dir, 'clusters.csv'))
        results.to_json(os.path.join(output_dir, 'clusters.json'))
        print(f"   ✓ Exported CSV and JSON")
        
        # Verify all outputs
        assert all(os.path.exists(p) for p in grid_paths.values())
        assert all(os.path.exists(p) for p in folder_paths.values())
        assert os.path.exists(os.path.join(output_dir, 'clusters.csv'))
        assert os.path.exists(os.path.join(output_dir, 'clusters.json'))
        
        print("\n6. Summary:")
        print(f"   ✓ Grids: {len(grid_paths)}")
        print(f"   ✓ Folders: {len(folder_paths)}")
        print(f"   ✓ Exports: 2 files")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n✅ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("VISUALIZATION TEST SUITE")
    print("="*60)
    
    results = []
    
    results.append(("GridVisualizer", test_grid_visualizer()))
    results.append(("Create Cluster Grids", test_create_cluster_grids()))
    results.append(("Results.create_grids()", test_results_create_grids()))
    results.append(("Results.create_cluster_folders()", test_create_cluster_folders()))
    results.append(("Complete Workflow", test_complete_workflow()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:40s} {status}")
    
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