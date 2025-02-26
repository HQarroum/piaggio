import os
import tempfile

from PIL import Image
from tqdm import tqdm
from scenedetect import ContentDetector, SceneManager, open_video
from scenedetect.video_splitter import is_ffmpeg_available
from scenedetect.scene_manager import save_images

# A list of supported image file extensions.
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

def load_images_from_directory(directory: str) -> list:
    """
    Load all images from the given directory.
    :param directory: The directory containing images.
    :return: A list of tuples (image, file_path).
    """
    images = []

    # Check if the directory exists and is not empty.
    if not os.path.isdir(directory):
        raise ValueError(f'Directory {directory} not found.')
    if not os.listdir(directory):
        raise ValueError(f'Directory {directory} is empty.')

    # List all files in the directory and filter out non-image files.
    image_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    # Load the images in memory.
    for image_file in tqdm(image_files, desc='Loading images'):
        image = Image.open(image_file).convert('RGB')
        images.append((image, image_file))
    return images


def load_images_from_video(video_path: str) -> list:
  """
  Load images from a video file.
  :param video_path: The path to the video file.
  :return: A list of scenes containing images.
  """
  if not os.path.isfile(video_path):
    raise ValueError(f'Video file {video_path} not found.')
  if not is_ffmpeg_available():
    raise ValueError('FFmpeg not found. Please install FFmpeg to extract images from videos.')
  
  # Open the video file.
  video = open_video(video_path)

  # Extract frames from the video.
  scene_manager = SceneManager()
  scene_manager.add_detector(ContentDetector(min_scene_len=15))
  scene_manager.detect_scenes(video=video, show_progress=True)
  scene_list = scene_manager.get_scene_list()
  if not scene_list:
    return []

  # Create a temporary directory.
  temp_dir = tempfile.mkdtemp()
  save_images(
      scene_list=scene_list,
      video=video,
      threading=True,
      output_dir=temp_dir,
      image_extension='jpg',
      show_progress=True
  )

  # Load back the frames in memory.
  return load_images_from_directory(temp_dir)
