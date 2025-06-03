import numpy as np
import sys
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as colors
from sklearn.cluster import KMeans, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import cv2
from skimage import color

def _cluster(folder_location, colorspace, algo, n, show=True):
    global model
    directory = folder_location
    results = np.empty(3, int)
    for filename in glob.glob(directory + '/*.png'):
        with open(filename):
            #read and reshape image
            img = cv2.imread(filename)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            if colorspace is 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pixels = img.reshape(-1, 3)

            nonblack_pixels = []
            for p in pixels:
                if p[0] != 0 or p[1] != 0 or p[2] != 0:
                    nonblack_pixels.append(p)
            nonblack_pixels = np.array(nonblack_pixels)

            #kmeans clustering
            kmeans = KMeans(n_clusters=4)
            kmeans.fit(nonblack_pixels)

            #get dominant color
            counts = np.bincount(kmeans.labels_)
            dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
            dominant_color = np.array(dominant_color.round(0).astype(int))

            #append array
            results = np.append(results, dominant_color)
    results = np.reshape(results, (-1, 3))

    match algo:
        case "kmeans":
            model = KMeans(n_clusters=n)
        case "gaussian_mixture":
            model = GaussianMixture(n_components=n,covariance_type='diag',reg_covar=1)#,reg_covar=2)
        case "agglom":
            if show:
                clusters = hierarchy.linkage(results, method="ward")
                plt.figure(figsize=(8, 6))
                dendrogram = hierarchy.dendrogram(clusters)
                # Plotting a horizontal line based on the first biggest distance between clusters
                plt.axhline(150, color='red', linestyle='--');
                # Plotting a horizontal line based on the second biggest distance between clusters
                plt.axhline(100, color='crimson');
            model = AgglomerativeClustering(n_clusters=n)
        case "dbscan":
            model = DBSCAN(eps=params_dict['eps'], min_samples=params_dict['min_samples'])#, #algorithm='ball_tree')  # , metric='manhattan')

    labels = model.fit_predict(results)
    print(labels)
    if show:
        # Define distinct markers for each cluster
        match n:
            case 2:
                markers = ['o', 's']  # 'o' is circle, 's' is square, 'D' is diamond, add more if needed
            case 3:
                markers = ['o', 's', 'D']  # 'o' is circle, 's' is square, 'D' is diamond, add more if needed

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot each cluster using its distinct marker
        for idx, marker in enumerate(markers):
            subset = results[labels == idx]
            colors = subset / 255
            if colorspace is "hsv":
                colors = color.hsv2rgb(colors)
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=colors, marker=marker, label=f'Cluster {idx}')
        if colorspace is "rgb":
            ax.set_xlabel('Red')
            ax.set_ylabel('Green')
            ax.set_zlabel('Blue')
        if colorspace is "hsv":
            ax.set_xlabel('Hue')
            ax.set_ylabel('Saturation')
            ax.set_zlabel('Value')
        ax.legend()
        plt.show()
    return results

_cluster(folder_location='D:/anole_data\segments',
         colorspace='hsv',algo='agglom',n=2,show=True)
