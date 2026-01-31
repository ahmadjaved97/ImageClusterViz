"""
Hashing utilities for duplicate detection.
"""

import hashlib
import numpy as np
from PIL import Image
from typing import Union

def compute_crypto_hash(
    image_path,
    algorithm: 'md5'
):
    """
    Compute cryptographic hash of image file.
    """

    if algorithm  == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    with open(image_path, 'rb') as f:
        # Read in chunks for memory efficiency

        while chunk := f.read(8192):
            hasher.update(chunk)
        
    return hasher.hexdigest()

def compute_average_hash(
    image,
    hash_size=8
):
    """
    Compute average hash (aHash).
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize and convert to grayscale
    image = image.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image).flatten()

    # Compare to mean
    avg = pixels.mean()
    hash_bits = pixels > avg

    return hash_bits.astype(np.uint8)


def compute_difference_hash(
    image,
    hash_size=8
):
    """
    Compute difference hash (dHash)
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize to hashsize + 1 for width gradient
    image = image.convert('L').resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS
    )

    pixels = np.array(image)

    # Compute horizontal gradient
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_bits = diff.flatten()

    return hash_bits.astype(np.uint8)

def compute_perceptual_hash(
    image,
    hash_size=8,
    highfreq_factor=4
):
    """
    Compute perceptual hash (phash) using DCT.
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_size = hash_size * highfreq_factor
    image = image.convert('L').resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    # Compute DCT
    try:
        from scipy.fftpack import dct
        dct_coef = dct(dct(pixels, axis=0), axis=1)
    except ImportError:
        # Fallback to numpy FFT (less accurate)
        dct_coef = np.fft.fft2(pixels).real
    
    # Extract low frequencies
    dct_low = dct_coef[:hash_size, :hash_size]

    # Compare to median
    median = np.median(dct_low)
    hash_bits = dct_low > median

    return hash_bits.flatten().astype(np.uint8)


def compute_wavelet_hash(
    image,
    hash_size=8,
    mode='haar'
):
    """
    Compute wavelet hash (wHash) using DWT.
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('L').resize(
        (hash_size * 2, hash_size * 2),
        Image.Resampling.LANCZOS
    )

    pixels = np.array(image, dtype=np.float32)

    try:
        import pywt

        # Apply 2D wavelet transform
        coeffs = pywt.dwt2(pixels, mode)

        # Use LL (approximation) coefficients
        ll, (lh, hl, hh) = coeffs

        # Compare to median
        median = np.median(ll)
        hash_bits = ll > median
    
    except ImportError:
        # Fallback : simple downsampling
        from scipy.ndimage import zoom
        downsampled = zoom(pixels, 0.5)
        median = np.median(downsampled)
        hash_bits = downsampled > median
    
    return hash_bits.flatten().astype(np.uint8)


def hamming_distance(hash1, hash2):
    """
    Compute hamming distance between two binary hashes.
    """

    return np.sum(hash1 != hash2)

def normalized_hamming_distance(hash1, hash2):
    """
    Compute normalized hamming distance (0-1 scale).
    """

    return hamming_distance(hash1, hash2) / len(hash1)

def hamming_similarity(hash1, hash2):
    """
    Compute hamming similarity (0-1 scale).
    """

    return 1.0 - normalized_hamming_distance(hash1, hash2)