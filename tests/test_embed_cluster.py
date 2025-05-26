# tests/test_embed_cluster.py
from imageclusterviz import embed_dir, cluster_kmeans, make_grid
from PIL import Image
import tempfile


def test_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        # generate 6 coloured squares
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)] * 2):
            Image.new("RGB", (32, 32), color).save(f"{td}/{i}.png")
        vecs, _ = embed_dir(td, model="resnet50", device="cpu")
        labels = cluster_kmeans(vecs, 2)
        grid = make_grid(td, labels, cols=3, thumb=(16, 16))
        assert grid.size[0] == 68 and grid.size[1] == 42
