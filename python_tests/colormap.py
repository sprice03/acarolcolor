import numpy as np
import sys
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as colors
from scipy.stats import gaussian_kde


np.set_printoptions(threshold = sys.maxsize)
def visualizeSegs(folder_location, colorspace):
    directory = folder_location
    results = np.empty(3, int)
    for filename in glob.glob(directory + '/*.png'):
        with open(filename):
            # read and reshape image
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

            # kmeans clustering
            kmeans = KMeans(n_clusters=4)
            kmeans.fit(nonblack_pixels)

            # get dominant color
            counts = np.bincount(kmeans.labels_)
            dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
            dominant_color = np.array(dominant_color.round(0).astype(int))

            # append array
            results = np.append(results, dominant_color)
    results = np.reshape(results, (-1, 3))

    # display dominant color for each image
    # dominant_color_img = np.zeros((100, 100, 3), dtype = 'uint8')
    # dominant_color_img[:, :, :] = dominant_color
    # plt.imshow(dominant_color_img)
    # plt.show()

    print(results)
    if colorspace == 'rgb':
        #plot
        fig = plt.figure()

        r = results[:,0]
        g = results[:,1]
        b = results[:,2]

        results_color = results/255

        ax = plt.subplot(1, 1, 1, projection='3d')

        ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors = results_color,  marker = "o")

        r, g, b = [228, 57], [233, 47], [250, 0]
        ax.plot(r, g, b)
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")



    else:
        #plot hsv
        fig = plt.figure(figsize=(10,8))

        h = results[:, 0]
        s = results[:, 1]
        v = results[:, 2]

        results_color = colors.hsv_to_rgb(results/255)

        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=results_color, marker="o")
        ax.set_xlabel("Hue")
        ax.set_ylabel("Saturation")
        ax.set_zlabel("Brightness")
    # Init the KDE
    h = results[:, 0]
    s = results[:, 1]
    v = results[:, 2]
    data = np.vstack([h,s,v])
    kde = gaussian_kde(data)

    # Evaluate the KDE on the data
    density = kde(data)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize the colors based on the density
    norm = plt.Normalize(vmin=density.min(), vmax=density.max())
    color = plt.cm.viridis(norm(density))

    # Display the scatter plot
    mesh = ax.scatter(h, s, v, c=color, marker='o', edgecolors='w', s=50)

    # Add a colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    mappable.set_array(density)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.75)
    cbar.set_label('Density')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

visualizeSegs('D:/anole_data\segments/green', 'rgb')
