import os
import shutil
import argparse

from pathlib import Path
from clean import clean_images
from tqdm import tqdm
from embeddings import (
  get_image_embeddings,
  cluster_images,
  uniq_per_cluster
)
from processing import (
  load_images_from_directory,
  load_images_from_video
)
from plot import (
  plot_embeddings,
  plot_embeddings_with_images
)

parser = argparse.ArgumentParser()

# Image directory.
parser.add_argument(
  '-d', '--directory',
  help='The directory containing images to process.'
)

# Video file.
parser.add_argument(
  '-v', '--video',
  help='The video file to process.'
)

# DBSCAN epsilon value.
parser.add_argument(
  '-e', '--epsilon',
  help='The epsilon value for DBSCAN clustering.',
  type=float,
  default=0.2
)

# DBSCAN minimum samples.
parser.add_argument(
  '-s', '--min-samples',
  help='The minimum number of samples for DBSCAN clustering.',
  type=int,
  default=2
)

# DBSCAN metric.
parser.add_argument(
  '-t', '--metric',
  help='The distance metric for DBSCAN clustering.',
  default='cosine'
)

# Whether to plot the embeddings.
parser.add_argument(
  '-p', '--plot',
  help='Whether to plot the embeddings.',
  action='store_true'
)

# Whether to plot the embeddings with images.
parser.add_argument(
  '-i', '--plot-images',
  help='Whether to plot the embeddings with images.',
  action='store_true'
)

# The output for the results.
parser.add_argument(
  '-o', '--output-dir',
  help='The output directory for the results.'
)

# Parse the command-line arguments.
args = parser.parse_args()

# Check if main function.
if __name__ == '__main__':
  images = []

  dir_provided = args.directory is not None and Path(args.directory).is_dir()
  video_provided = args.video is not None and Path(args.video).is_file()
  output_dir = Path(args.output_dir or 'output')

  # Verify that the user provided either a directory or a video file.
  if not dir_provided and not video_provided:
    parser.error('Please provide either a directory or a video file.')

  # Create the output directory if it does not exist.
  output_dir.mkdir(parents=True, exist_ok=True)

  # Load images from the source.
  if dir_provided:
    images = load_images_from_directory(args.directory)
  else:
    images = load_images_from_video(args.video)
  
  # Clean the images dataset.
  images = clean_images(images)
  if len(images) < 1:
    raise ValueError('Not enough images to cluster.')

  # Generate embeddings
  embeddings = get_image_embeddings(images).cpu().numpy()

  # Cluster images
  labels = cluster_images(
    embeddings,
    eps=args.epsilon,
    min_samples=args.min_samples,
    metric=args.metric
  )

  # Filter images
  unique_images = uniq_per_cluster(images, labels)

  # Copy the selected images to the output directory.
  for image in tqdm(unique_images, desc="Copying images", unit="image"):
    image_name = os.path.basename(image)
    output_path = os.path.join(args.output_dir, image_name)
    shutil.copy(image, output_path)

  # Plot the embeddings if requested.
  if args.plot:
    plot_embeddings(embeddings, labels)
  elif args.plot_images:
    plot_embeddings_with_images(images, embeddings, labels)
