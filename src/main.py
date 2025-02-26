import os
import shutil

from pathlib import Path
from clean import clean_images
from args import parser
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

# Command-line arguments.
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

  # Get unique images from each cluster.
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
