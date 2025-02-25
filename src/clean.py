import cv2
import numpy as np
from tqdm import tqdm

def is_black_frame(pil_image, threshold=10, black_pixel_ratio=0.95):
    """
    Check if a given image is considered a "black frame" based on the proportion
    of very dark pixels.
    """
    image_np = np.array(pil_image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if image_np.ndim == 3 else image_np
    ratio = np.sum(gray_image < threshold) / gray_image.size
    return ratio > black_pixel_ratio


def is_technical_frame(pil_image, black_threshold=10, white_threshold=180, 
                       pixel_fraction_threshold=0.95, variance_threshold=500):
    """
    Identify if an image is a technical frame (black, white, or uniform).
    """
    image_np = np.array(pil_image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if image_np.ndim == 3 else image_np
    total_pixels = gray_image.size
    black_pixels = np.sum(gray_image < black_threshold)
    white_pixels = np.sum(gray_image > white_threshold)
    variance = np.var(gray_image)

    is_black = (black_pixels / total_pixels) > pixel_fraction_threshold
    is_white = (white_pixels / total_pixels) > pixel_fraction_threshold
    is_uniform = variance < variance_threshold

    return is_black or is_white or is_uniform


def is_blurry_image(pil_image, threshold=100.0):
    """
    Check if an image is blurry using the Laplacian variance method.
    Returns a tuple (variance, is_blurry).
    """
    image_np = np.array(pil_image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if image_np.ndim == 3 else image_np
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance, variance < threshold


def clean_images(images):
    """
    Clean a list of images by filtering out black, technical, and blurry frames.
    Parameters:
    images (list): A list of tuples (image, file_path) where image is a PIL image.
    
    Returns:
    list: A list of cleaned images.
    """
    cleaned_images = []
    for image, file_path in tqdm(images, desc="Cleaning images", unit="image"):
        if is_technical_frame(image):
            print(f"Skipping technical frame {file_path}")
        elif is_black_frame(image):
            print(f"Skipping black frame {file_path}")
        else:
            cleaned_images.append((image, file_path))

    return cleaned_images
