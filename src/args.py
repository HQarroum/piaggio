import argparse

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

# CLIP model.
parser.add_argument(
  '-m', '--model',
  help='The CLIP model to use for image embeddings.',
  default='ViT-B/32'
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
