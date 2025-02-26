import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def is_technical_frame(
    pil_image: Image,
    black_threshold=10,
    white_threshold=180, 
    pixel_fraction_threshold=0.95,
    variance_threshold=500
) -> bool:
    """
    Identify if an image is a technical frame (black, white, or uniform).
    :param pil_image: The input image as a PIL image.
    :param black_threshold: The threshold for black pixels.
    :param white_threshold: The threshold for white pixels.
    :param pixel_fraction_threshold: The fraction of pixels that should be black or white.
    :param variance_threshold: The threshold for the variance of the image.
    :return: True if the image is a technical frame, False otherwise.
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


def is_blurry_image(pil_image, threshold=100.0) -> tuple:
    """
    Check if an image is blurry using the Laplacian variance method.
    :param pil_image: The input image as a PIL image.
    :param threshold: The threshold value for the variance.
    :return: A tuple (variance, is_blurry).
    """
    image_np = np.array(pil_image)

    if image_np.ndim == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_np

    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance, variance < threshold


def clean_images(images: list) -> list:
    """
    Clean a list of images by filtering out technical frames.
    :images (list): A list of tuples (image, file_path) where image is a PIL image.
    :return: A list of cleaned images.
    """
    cleaned_images = []
    for image, file_path in tqdm(images, desc="Cleaning images", unit="image"):
        if not is_technical_frame(image):
            cleaned_images.append((image, file_path))
    return cleaned_images
