from enum import Enum
import cv2
from operator import attrgetter
from functools import partial
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

    @staticmethod
    def match(properties, des1, des2, kp1, kp2, min_distance=100):
        matches = properties.matcher.match(des1, des2)
        min_match_dist = min(min_distance,
                             min(matches, key=attrgetter('distance')).distance) if matches else min_distance
        return [m for m in matches
                if m.distance <= 3 * min_match_dist]


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


class FeatureDetector(Enum):
    SIFT = cv2.xfeatures2d.SIFT_create  # take less memory, resilient to scaling
    SURF = cv2.xfeatures2d.SURF_create  # = quicker SIFT resilient to rotation and blur, sensitive to change of viewpoint and illumination
    ORB = cv2.ORB_create  # = fusion and enhancement to FAST&BRIEF, resilient to rotation and scaling
    AKAZE = cv2.AKAZE_create  #
    BRISK = cv2.BRISK_create  # similar to ORB, different method
    KAZE = cv2.KAZE_create  #
    MSER = cv2.MSER_create  #
    AGAST = cv2.AgastFeatureDetector_create  #
    FAST = cv2.FastFeatureDetector_create  # fast, sensitive to noise, rotation and scaling, need threshold
    GFTT = cv2.GFTTDetector_create  # resilient to rotation, sensitive to scaling
    BLOB = cv2.SimpleBlobDetector_create  #
    HARRIS = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create  # sensitive to scaling
    # MSD     = cv2.xfeatures2d.MSDDetector_create    #
    STAR = cv2.xfeatures2d.StarDetector_create  # = CenSurE, use w/ BRIEF


class FeatureDescriptor(Enum):
    SIFT = cv2.xfeatures2d.SIFT_create  #
    SURF = cv2.xfeatures2d.SURF_create  #
    ORB = cv2.ORB_create  #
    AKAZE = cv2.AKAZE_create  #
    BRISK = cv2.BRISK_create  #
    KAZE = cv2.KAZE_create  #
    VGG = cv2.xfeatures2d.VGG_create  #
    DAISY = cv2.xfeatures2d.DAISY_create  #
    LATCH = cv2.xfeatures2d.LATCH_create  #
    LUCID = cv2.xfeatures2d.LUCID_create  #
    FREAK = cv2.xfeatures2d.FREAK_create  #
    BOOST = cv2.xfeatures2d.BoostDesc_create  #
    BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create  # take less memory


class Matcher(Enum):
    FLANN = 1
    BFM = 2


class MatchMethod(Enum):
    MATCH = partial(MatcherMethodDef.match)
    KNN = partial(MatcherMethodDef.knn_match)


class HomographyMethod(Enum):
    RANSAC = cv2.RANSAC
    LMEDS = cv2.LMEDS
    RHO = cv2.RHO


class PropertiesGenerator:

    @staticmethod
    def circular_generator(iter):
        while True:
            for item in iter:
                yield item

    def __init__(self):
        self.detector_choice = None
        self.detector = None
        self.descriptor_choice = None
        self.descriptor = None
        self.matcher_choice = None
        self.matcher = None
        self.matcher_method_choice = None
        self.homography_method_choice = None
        self.homography_method = None
        self.k = 2
        self.color = False
        with open('calibration.yaml') as f:
            loadeddict = yaml.load(f)
            self.camera_matrix = numpy.asarray(loadeddict.get('cameraMatrix'))
            self.dist_coeffs = numpy.asarray(loadeddict.get('distCoeffs'))

        self.detector_gen = self.circular_generator(FeatureDetector)
        self.descriptor_gen = self.circular_generator(FeatureDescriptor)
        self.matcher_gen = self.circular_generator(Matcher)
        self.match_method_gen = self.circular_generator(MatchMethod)
        self.homography_method_gen = self.circular_generator(HomographyMethod)
        self.update_detector()
        self.update_descriptor(True)
        self.update_matcher_method(True)
        self.update_matcher()
        self.update_homography_method()

    def update_detector(self):
        self.detector_choice = next(self.detector_gen)
        self.detector = self.detector_choice.value()

    def update_descriptor(self, init=False):
        self.descriptor_choice = next(self.descriptor_gen)
        self.descriptor = self.descriptor_choice.value()
        self.color = self.descriptor_choice == FeatureDescriptor.LUCID
        if not init:
            self.matcher = self.get_correct_matcher()

    def update_matcher(self):
        self.matcher_choice = next(self.matcher_gen)
        self.matcher = self.get_correct_matcher()

    def update_matcher_method(self, init=False):
        self.matcher_method_choice = next(self.match_method_gen)
        if not init and self.matcher_choice == Matcher.BFM:
            self.matcher = cv2.BFMatcher.create(self.descriptor.defaultNorm(),
                                                crossCheck=self.matcher_method_choice == MatchMethod.MATCH)

    def update_homography_method(self):
        self.homography_method_choice = next(self.homography_method_gen)
        self.homography_method = self.homography_method_choice.value

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

        if self.matcher_choice == Matcher.BFM:
            matcher = cv2.BFMatcher.create(distance_measurement,
                                           crossCheck=self.matcher_method_choice == MatchMethod.MATCH)
        elif self.matcher_choice == Matcher.FLANN:
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return None

        return matcher
