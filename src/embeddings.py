import torch
import numpy as np
import clip

from args import parser
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# Command-line arguments.
args = parser.parse_args()

# Select the appropriate device to run the model on
# based on the availability of CUDA and MPS.
device = 'cuda' if torch.cuda.is_available() else \
  ('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the CLIP model and pre-processing function.
model, processor = clip.load(args.model, device=device)

def get_image_embeddings(images: list) -> torch.Tensor:
    """
    Create embeddings for a list of images using a pre-trained CLIP model.
    Processes images in batch for efficiency.
    :param images: List of PIL images to process.
    :return: Torch tensor of embeddings.
    """
    image_tensors = [
        processor(image)
            .unsqueeze(0) for image, _ in tqdm(
                images,
                desc='Creating embeddings',
                unit='image'
            )
    ]
    batch = torch.cat(image_tensors, dim=0).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(batch)
    return embeddings


def uniq_per_cluster(images: list, labels: np.ndarray) -> list:
    """
    Filter images to retain one image from each cluster.
    :param images: The images to filter.
    :param labels: The cluster labels for the images.
    :return: A list of unique images
    """
    unique_images = []
    for label in np.unique(labels):
      indices = np.where(labels == label)[0]
      selected_image = images[indices[0]][1]  # Taking the first image from each cluster
      unique_images.append(selected_image)
    return unique_images


def cluster_images(
    embeddings,
    eps=0.20,
    min_samples=2,
    metric='cosine'
) -> np.ndarray:
    """
    Cluster images using DBSCAN based on their embeddings.
    :param embeddings: The image embeddings (as a torch tensor or NumPy array).
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :param metric: The distance metric to use.
    :return: Cluster labels as a NumPy array.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    ).fit(embeddings)
    
    return clustering.labels_
