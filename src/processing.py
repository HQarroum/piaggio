import os
import tempfile

from PIL import Image, ImageFile
from tqdm import tqdm
from scenedetect import ContentDetector, SceneManager, open_video
from scenedetect.video_splitter import is_ffmpeg_available
from scenedetect.scene_manager import save_images

# Allow loading truncated images.
ImageFile.LOAD_TRUNCATED_IMAGES=True

# A list of supported image file extensions.
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def load_images_from_directory(directory):
    """
    Load all images from the given directory.
    Returns a list of tuples (PIL image, file_path).
    """
    images = []
    if not os.path.isdir(directory):
        raise ValueError(f'Directory {directory} not found.')
    if not os.listdir(directory):
        raise ValueError(f'Directory {directory} is empty.')

    image_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    for image_file in tqdm(image_files, desc='Loading images'):
        image = Image.open(image_file).convert('RGB')
        images.append((image, image_file))
    return images


def load_images_from_video(video_path) -> list:
  """
  Load images from a video file.
  :param video_path: The path to the video file.
  :return: A list of scenes containing images.
  """
  if not os.path.isfile(video_path):
    raise ValueError(f'Video file {video_path} not found.')
  if not is_ffmpeg_available():
    raise ValueError('FFmpeg not found. Please install FFmpeg to extract images from videos.')
  
  # Open the video file; this returns a VideoStream instance.
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
  print(f"Extracting frames to {temp_dir}")
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
