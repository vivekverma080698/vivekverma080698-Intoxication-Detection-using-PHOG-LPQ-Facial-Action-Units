import math

import numpy as np
import cv2
from skimage import feature
from scipy import ndimage


class PHogFeatures():
    """
    Computes a PHOG descriptor of an image. This code was converted from Matlab to Python. The main difference is
    that it uses a scikit canny implementation and no ROI (the region of interest is the whole image).
    The original Matlab implementation is based here: [http://www.robots.ox.ac.uk/~vgg/research/caltech/phog.html]
    """

    def __init__(self):
        pass

    def get_features(self, image_path, bins=8, angle=360., pyramid_levels=3):
        """
        Returns a feature vector containing a PHOG descriptor of a whole image.

        :param image_path: Absolute path to an image
        :param bins: Number of (orientation) bins on the histogram (optimal: 20)
        :param angle: 180 or 360 (optimal: 360)
        :param pyramid_levels: Number of pyramid levels (optimal: 3)
        :return:
        """

        feature_vec = self.phog(image_path, bins, angle, pyramid_levels)
        feature_vec = feature_vec.T[0]  # Transpose vector, take the first array
        return feature_vec

    def phog(self, image_path, bin, angle, pyramid_levels):
        """
        Given and image I, phog computes the Pyramid Histogram of Oriented
        Gradients over L pyramid levels and over a Region Of Interest.

        :param image_path: Absolute path to an image of size MxN (Color or Gray)
        :param bin: Number of (orientation) bins on the histogram
        :param angle: 180 or 360
        :param pyramid_levels: Number of pyramid levels
        :return: Pyramid histogram of oriented gradients
        """

        grayscale_img = cv2.imread(image_path, 0)  # 0 converts it to grayscale

        bh = np.array([])
        bv = np.array([])
        if np.sum(np.sum(grayscale_img)) > 100.:
            # Matlab The default sigma is sqrt(2); the size of the filter is chosen automatically, based on sigma.
            # Threshold is applied automatically - the percentage is a bit different than in Matlab's implementation:
            # low_threshold: 10%
            # high_threshold: 20%
            edges_canny = feature.canny(grayscale_img, sigma=math.sqrt(2))
            [GradientY, GradientX] = np.gradient(np.double(grayscale_img))
            GradientYY = np.gradient(GradientY)[1]  # Take only the first matrix
            Gr = np.sqrt((GradientX * GradientX + GradientY * GradientY))

            index = GradientX == 0.
            index = index.astype(int)  # Convert boolean array to an int array
            GradientX[np.where(index > 0)] = np.power(10, 5)
            YX = GradientY / GradientX

            if angle == 180.:
                angle_values = np.divide((np.arctan(YX) + np.pi / 2.) * 180., np.pi)
            if angle == 360.:
                angle_values = np.divide((np.arctan2(GradientY, GradientX) + np.pi) * 180., np.pi)

            [bh, bv] = self.bin_matrix(angle_values, edges_canny, Gr, angle, bin)
        else:
            bh = np.zeros(image_path.shape[0], image_path.shape[1])
            bv = np.zeros(image_path.shape[0], image_path.shape[1])

        # Don't consider a roi, take the whole image instead
        bh_roi = bh
        bv_roi = bv
        p = self.phog_descriptor(bh_roi, bv_roi, pyramid_levels, bin)

        return p

    def bin_matrix(self, angle_values, edge_image, gradient_values, angle, bin):
        """
        Computes a Matrix (bm) with the same size of the image where
        (i,j) position contains the histogram value for the pixel at position (i,j)
        and another matrix (bv) where the position (i,j) contains the gradient
        value for the pixel at position (i,j)

        :param angle_values: Matrix containing the angle values
        :param edge_image: Edge Image
        :param gradient_values: Matrix containing the gradient values
        :param angle: 180 or 360
        :param bin: Number of bins on the histogram
        :return: bm - Matrix with the histogram values
                bv - Matrix with the gradient values (only for the pixels belonging to and edge)
        """

        # 8-orientations/connectivity structure (Matlab's default is 8 for bwlabel)
        structure_8 = [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]

        [contorns, n] = ndimage.label(edge_image, structure_8)
        X = edge_image.shape[1]
        Y = edge_image.shape[0]
        bm = np.zeros((Y, X))
        bv = np.zeros((Y, X))
        nAngle = np.divide(angle, bin)
        for i in np.arange(1, n + 1):
            [posY, posX] = np.nonzero(contorns == i)
            posY = posY + 1
            posX = posX + 1
            for j in np.arange(1, (posY.shape[0]) + 1):
                pos_x = posX[int(j) - 1]
                pos_y = posY[int(j) - 1]
                b = np.ceil(np.divide(angle_values[int(pos_y) - 1, int(pos_x) - 1], nAngle))
                if b == 0.:
                    bin = 1.
                if gradient_values[int(pos_y) - 1, int(pos_x) - 1] > 0:
                    bm[int(pos_y) - 1, int(pos_x) - 1] = b
                    bv[int(pos_y) - 1, int(pos_x) - 1] = gradient_values[int(pos_y) - 1, int(pos_x) - 1]

        return [bm, bv]

    def phog_descriptor(self, bh, bv, pyramid_levels, bin):
        """
        Computes Pyramid Histogram of Oriented Gradient over an image.

        :param bh: Matrix of bin histogram values
        :param bv: Matrix of gradient values
        :param pyramid_levels: Number of pyramid levels
        :param bin: Number of bins
        :return: Pyramid histogram of oriented gradients (phog descriptor)
        """

        p = np.empty((0, 1), dtype=int)  # dtype=np.float64? # vertical size 0, horizontal 1

        for b in np.arange(1, bin + 1):
            ind = bh == b
            ind = ind.astype(int)  # convert boolean array to int array
            sum_ind = np.sum(bv[np.where(ind > 0)])
            p = np.append(p, np.array([[sum_ind]]), axis=0)  # append the sum horizontally to empty p array

        cella = 1
        for l in np.arange(1, pyramid_levels + 1):  # defines a range (from, to, step)
            x = np.fix(np.divide(bh.shape[1], 2. ** l))
            y = np.fix(np.divide(bh.shape[0], 2. ** l))
            xx = 0
            yy = 0
            while xx + x <= bh.shape[1]:
                while yy + y <= bh.shape[0]:
                    bh_cella = np.array([])
                    bv_cella = np.array([])
                    bh_cella = bh[int(yy + 1.) - 1:int(yy + y), int(xx + 1.) - 1:int(xx + x)]
                    bv_cella = bv[int(yy + 1.) - 1:int(yy + y), int(xx + 1.) - 1:int(xx + x)]

                    for b in np.arange(1, bin + 1):
                        ind = bh_cella == b
                        ind = ind.astype(int)  # convert boolean array to int array
                        sum_ind = np.sum(bv_cella[np.where(ind > 0)])
                        p = np.append(p, np.array([[sum_ind]]), axis=0)  # append the sum horizontally to p

                    yy = yy + y

                cella = cella + 1.
                yy = 0.
                xx = xx + x

        if np.sum(p) != 0:
            p = np.divide(p, np.sum(p))

        return p


# if __name__ == '__main__':
#     phog = PHogFeatures()
#
#     image_path = "./frame11.jpg"
#     print len(phog.get_features(image_path))