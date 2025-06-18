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


def getcontour(folder_location):
    directory = folder_location
    filenames = []
    print(directory)
    i = 0
    for filename in glob.glob(directory + '/masks/*.png'):
        with open(filename):
            mask = cv2.imread(filename)
            ids = filename.split('/')
            ids = ids[2].split('\\')
            filebase = ids.replace("_mask", "")
            filebase = filebase.replace(".png", ".jpg")
            image = cv2.imread(filebase)
            ids = ids[1].split(' ')
            ids = ids[0]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # edge detection
            edges = cv2.Canny(gray, 100, 200)
            # find contours
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, (255, 255, 255), 20)
            #exclude mask from contour
            bitwise = cv2.bitwise_xor(image, mask)
            original = cv2.imread(directory + 'all_images/' + filebase)
            bitwise2 = cv2.bitwise_and(original, bitwise)
            cv2.imwrite(directory + 'edges/' + ids + '.png', bitwise2)
        i = i + 1


def greenindex(folder_location, colorspace, colormethod, clusters):
    directory = folder_location
    results = np.empty(3, int)
    filenames = []
    location = folder_location.split("/")
    location = location[3]
    colorout = np.array(['id', location + '_color'])
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
            #get green index
            greenindex = dominant_color[1] / (dominant_color[0] + dominant_color[1] + dominant_color[2])
            colorresult = [ids, greenindex]
            colorout = np.append(colorout, colorresult)
        i = i + 1
    results = np.reshape(results, (-1, 3))
    colorout = np.reshape(colorout, (-1, 2))
    print(colorout)
    df = pd.DataFrame(colorout)
    df.to_csv(folder_location + "/" + location + "_" + "greenindex.csv", header=False, index=False)


getcontour('F:/a_carol_data/alachua_gainesville')
greenindex('F:/a_carol_data/alachua_gainesville/segments', 'rgb', 'kmeans', 4)
greenindex('F:/a_carol_data/alachua_gainesville/edges', 'rgb', 'kmeans', 4)
