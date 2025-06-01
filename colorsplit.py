import numpy as np
import sys
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)


def visualizeSegs(folder_location, colorspace, colormethod, clusters):
    directory = folder_location
    results = np.empty(3, int)
    colorout = np.array(['id', 'color'])
    filenames = []
    location = folder_location.split("/")
    location = location[2]
    i = 0
    for filename in glob.glob(directory + '/*.png'):
        with open(filename):
            ids = filename.split("\\")
            ids = ids[1].split()
            ids = ids[0]
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

            # kmeans clustering or mean color, then get dominant color
            if colormethod is 'kmeans':
                kmeans = KMeans(n_clusters=clusters)
                kmeans.fit(nonblack_pixels)
                counts = np.bincount(kmeans.labels_)
                dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
                dominant_color = np.array(dominant_color.round(0).astype(int))
            if colormethod is 'mean':
                dominant_color = np.mean(nonblack_pixels, axis=0).round(0)
                print(dominant_color)

            # append array
            results = np.append(results, dominant_color)
            filenames.append(filename)

            # identify dominant color
            if dominant_color[0] > dominant_color[1] + 5:
                # location = "D:/anole_data\segments/brown/" + str(i) + ".jpg"
                # cv2.imwrite(location, img)
                colorresult = [ids, 'brown']
                colorout = np.append(colorout, colorresult)
                if 'green' in directory:
                    plt.subplot(1, 2, 1)
                    dominant_color_img = np.zeros((100, 100, 3), dtype='uint8')
                    dominant_color_img[:, :, :] = dominant_color
                    plt.imshow(dominant_color_img)

                    plt.subplot(1, 2, 2)
                    plt.imshow(img)

                    plt.show()
            elif dominant_color[1] > dominant_color[0] + 2:
                # location = "D:/anole_data\segments/green/" + str(i) + ".jpg"
                # cv2.imwrite(location, img)
                colorresult = [ids, 'green']
                colorout = np.append(colorout, colorresult)
                if 'brown' in directory:
                    plt.subplot(1, 2, 1)
                    dominant_color_img = np.zeros((100, 100, 3), dtype='uint8')
                    dominant_color_img[:, :, :] = dominant_color
                    plt.imshow(dominant_color_img)

                    plt.subplot(1, 2, 2)
                    plt.imshow(img)

                    plt.show()
            # if this is not a sorted test set, deactivate this part of code
            else:
                colorresult = [ids, 'unclear']
                colorout = np.append(colorout, colorresult)
        i = i + 1
    results = np.reshape(results, (-1, 3))
    colorout = np.reshape(colorout, (-1, 2))
    print(colorout)
    df = pd.DataFrame(colorout)
    # df.to_csv(folder_location + "/" + location + "_" + "colorout.csv", header=False, index=False)

    # sum the occurance of brown, green, and unclear observations
    brown = np.sum(colorout == 'brown')
    green = np.sum(colorout == 'green')
    unclear = np.sum(colorout == 'unclear')
    # determine accuracy
    if 'brown' in directory:
        accuracy = brown / (brown + green)
    if 'green' in directory:
        accuracy = green / (brown + green)

    print(accuracy, unclear / (i + 1))
    print(directory)


visualizeSegs('D:/anole_data/segments/green', 'rgb', 'kmeans', 4)
