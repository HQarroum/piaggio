import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray):
    """
    Plot embeddings with PCA to 2D to visualize clusters.
    :param embeddings: The image embeddings to plot.
    :param labels: The cluster labels for the embeddings.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', marker='o')
    plt.colorbar()
    plt.title("PCA of Image Embeddings Colored by DBSCAN Labels")
    plt.show()


def plot_embeddings_with_images(
    images: list,
    embeddings: np.ndarray,
    labels: np.ndarray
):
    """
    Plot images with PCA to 2D, displaying images at their cluster positions.
    :param images: The PIL images to plot.
    :param embeddings: The image embeddings to plot.
    :param labels: The cluster labels for the embeddings.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    _, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap='viridis',
        marker='o',
        s=100
    )

    # Create space for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Add a colorbar in the space just created
    plt.colorbar(scatter, cax=cax, label='Cluster label')

    for i, (image, _) in enumerate(images): 
        img = np.array(image.resize((50, 50)))
        imgbox = OffsetImage(img, zoom=1.0)
        ab = AnnotationBbox(imgbox, (reduced[i, 0], reduced[i, 1]), frameon=False)
        ax.add_artist(ab)

    plt.title('PCA of Image Embeddings with Images', fontsize=15)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()
