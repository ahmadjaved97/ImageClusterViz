from pathlib import Path
from typing import Sequence, Tuple, List
import math
import numpy as np
from PIL import Image
import cv2

__all__ = ["create_image_grid", "make_grid"]


# ----------------------------------------------------------------------
# Core helper
# ----------------------------------------------------------------------
def create_image_grid(
    image_list: Sequence[np.ndarray | Image.Image],
    num_rows: int | None = None,
    num_cols: int | None = None,
    *,
    image_size: Tuple[int, int] = (100, 100),
    space_color: Tuple[int, int, int] = (255, 255, 255),
    space_width: int = 10,
) -> np.ndarray:
    """
    Arrange *image_list* into a grid with optional spacing & row/col labels.

    Returns
    -------
    grid_bgr : np.ndarray (H, W, 3) – always BGR uint8 so it can be
               saved with cv2.imwrite or wrapped in PIL later.
    """
    # --- normalise inputs ------------------------------------------------
    # convert PIL → ndarray BGR
    normalized: List[np.ndarray] = []
    for im in image_list:
        if isinstance(im, Image.Image):
            im = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(im, np.ndarray):
            if im.ndim == 2:  # gray → BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 3:  # assume already BGR/RGB – we won't swap
                pass
            else:
                raise ValueError("Expected H×W×3 or H×W gray image array.")
        else:
            raise TypeError(f"Unsupported image type: {type(im)}")
        normalized.append(im)

    # --- infer grid shape ------------------------------------------------
    if num_rows is None and num_cols is None:
        num_images = len(normalized)
        num_cols = int(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_cols)
    elif num_rows is None:
        num_rows = math.ceil(len(normalized) / num_cols)
    elif num_cols is None:
        num_cols = math.ceil(len(normalized) / num_rows)
    # now both are ints
    assert num_rows * num_cols >= len(
        normalized
    ), "Rows/cols insufficient for number of images"

    # --- create background canvas ---------------------------------------
    grid_w = num_cols * (image_size[0] + space_width) - space_width
    grid_h = num_rows * (image_size[1] + space_width) - space_width
    canvas = np.full((grid_h, grid_w, 3), space_color, dtype=np.uint8)

    # --- paste images ----------------------------------------------------
    for idx, im in enumerate(normalized):
        im_res = cv2.resize(im, image_size)
        r, c = divmod(idx, num_cols)
        x0 = c * (image_size[0] + space_width)
        y0 = r * (image_size[1] + space_width)
        canvas[y0 : y0 + image_size[1], x0 : x0 + image_size[0]] = im_res

    # --- draw row / column numbers --------------------------------------
    for r in range(num_rows):
        y_txt = r * (image_size[1] + space_width) + 20
        cv2.putText(
            canvas,
            str(r + 1),
            (5, y_txt),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    for c in range(num_cols):
        x_txt = c * (image_size[0] + space_width) + 5
        cv2.putText(
            canvas,
            str(c + 1),
            (x_txt, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas  # BGR ndarray


# ----------------------------------------------------------------------
# Back-compat wrapper (returns PIL.Image)
# ----------------------------------------------------------------------
def make_grid(
    img_dir: str | Path,
    labels: Sequence[int],
    *,
    cols: int | None = None,
    thumb: Tuple[int, int] = (100, 100),
    space_color: Tuple[int, int, int] = (255, 255, 255),
    space_width: int = 10,
) -> Image.Image:
    """
    Convenience wrapper that loads all images from *img_dir* in the order
    specified by *labels* (cluster order) and returns a PIL Image.
    """
    paths = sorted(Path(img_dir).iterdir())
    assert len(paths) == len(labels), "labels & images mismatch"

    # order images by cluster label then filename
    sort_idx = np.lexsort((np.arange(len(paths)), labels))
    images = [Image.open(paths[i]).convert("RGB") for i in sort_idx]

    grid_bgr = create_image_grid(
        images,
        num_cols=cols,
        image_size=thumb,
        space_color=space_color,
        space_width=space_width,
    )
    return Image.fromarray(cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB))
