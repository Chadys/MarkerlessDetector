from enum import Enum
import cv2
import numpy
import yaml


class MatcherMethodDef:

    @staticmethod
    def knn_match(properties, des1, des2, kp1, kp2, nn_ratio=0.75):
        # Match features from each image
        if len(kp1) < properties.k or len(kp2) < properties.k:
            return []

        matches = properties.matcher.knnMatch(des1, des2, k=properties.k)
        # store only the good matches as per Lowe's ratio test.
        return [m[0] for m in matches
                if len(m) == 2
                and m[0].distance < nn_ratio * m[1].distance]


class FlannAlgorithm(Enum):
    FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    FLANN_INDEX_COMPOSITE = 3
    FLANN_INDEX_KDTREE_SINGLE = 4
    FLANN_INDEX_HIERARCHICAL = 5
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_SAVED = 254
    FLANN_INDEX_AUTOTUNED = 255
    LINEAR = 0
    KDTREE = 1
    KMEANS = 2
    COMPOSITE = 3
    KDTREE_SINGLE = 4
    SAVED = 254
    AUTOTUNED = 255


class PropertiesGenerator:

    def __init__(self):
        self.detector = cv2.KAZE_create()
        self.descriptor = cv2.xfeatures2d.SIFT_create()
        self.matcher_method_choice = MatcherMethodDef.knn_match
        self.matcher = self.get_correct_matcher()
        self.homography_method = cv2.RANSAC

        self.k = 2

        with open('calibration.yaml') as f:
            loadeddict = yaml.load(f)
            self.camera_matrix = numpy.asarray(loadeddict.get('cameraMatrix'))
            self.dist_coeffs = numpy.asarray(loadeddict.get('distCoeffs'))

    def get_correct_matcher(self):
        search_params = dict(checks=100)
        distance_measurement = self.descriptor.defaultNorm()
        if distance_measurement == cv2.NORM_HAMMING or distance_measurement == cv2.NORM_HAMMING2:
            index_params = dict(algorithm=FlannAlgorithm.FLANN_INDEX_LSH.value,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)
        else:
            index_params = dict(algorithm=FlannAlgorithm.KDTREE.value, trees=5)

        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        return matcher
