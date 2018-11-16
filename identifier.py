# references used :
# https://github.com/dmartinalbo/image-matching
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
import sys
import argparse
import numpy as np
import itertools
import cv2
from identifier_properties import PropertiesGenerator

MIN_MATCH_COUNT = 10


class Template:
    def __init__(self, name, img, kp, des):
        self.name = name
        self.img = img
        self.kp = kp
        self.des = des


class Identifier:
    def __init__(self, template_names):
        self.properties = PropertiesGenerator()
        self.error_text = ''
        self.templates = []
        self.load_template(template_names)

    def load_template(self, template_names):
        for name in template_names:
            print('Loading template image {}'.format(name))
            img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            print('  Calculating features ...')
            kp, des = self.calculate_feature_points(img)
            if des.size > 0:
                self.templates.append(Template(name, img, kp, des))
            img2 = None
            img = cv2.drawKeypoints(img, kp, img2)
            self.display_img(name, img)

    def update_template(self):
        for template in self.templates:
            template.kp, template.des = self.calculate_feature_points(template.img)
            img2 = None
            img = cv2.drawKeypoints(template.img, template.kp, img2)
            self.display_img(template.name, img)

    def calculate_feature_points(self, img):
        # Find the keypoints and descriptors using features
        kp = self.properties.detector.detect(img, None)
        try:
            kp, des = self.properties.descriptor.compute(img, kp)
            self.error_text = ''
        except cv2.error:
            self.error_text = 'Invalid Detector/Descriptor combination : '
            return None, None
        return kp, des

    def find_match(self, des1, des2, kp1, kp2, extra_arg=None):
        if extra_arg is None:
            return self.properties.matcher_method_choice.value(self.properties, des1, des2, kp1, kp2)
        return self.properties.matcher_method_choice.value(self.properties, des1, des2, kp1, kp2, extra_arg)

    @staticmethod
    def template_classif(good_matches):
        return max(range(len(good_matches)), key=lambda i: len(good_matches[i]))

    @staticmethod
    def is_valid_square(square_pts):
        pts_pairs = itertools.combinations(square_pts, 2)
        # check if any two points are equal
        for pair in pts_pairs:
            if round(pair[0][0][0], 0) == round(pair[1][0][0], 0) \
                    and round(pair[0][0][1], 0) == round(pair[1][0][1], 0):
                return False
        #check if 3 points are aligned
        if Identifier.collinear_3_of_4(square_pts[0][0], square_pts[1][0], square_pts[2][0], square_pts[3][0]):
            return False
        # TODO check if points form a converx figure
        return True

    @staticmethod
    def collinear_3_of_4(p1, p2, p3, p4, eps=0.0005):
        # (p1, p2, p3) are collinear if and only if
        #     abs( (p2.x-p1.x)*(p3.y-p1.y) -
        #          (p3.x-p1.x)*(p2.y-p1.y) ) <= eps
        (x12, y12) = (p2[0] - p1[0], p2[1] - p1[1])
        (x13, y13) = (p3[0] - p1[0], p3[1] - p1[1])
        (x14, y14) = (p4[0] - p1[0], p4[1] - p1[1])
        (x23, y23) = (p3[0] - p2[0], p3[1] - p2[1])
        (x24, y24) = (p4[0] - p2[0], p4[1] - p2[1])
        # Test each unique triplet.
        # 4 choose 3 = 4 triplets: 123, 124, 134, 234
        return (abs(x12 * y13 - x13 * y12) < eps)\
            or (abs(x12 * y14 - x14 * y12) < eps)\
            or (abs(x13 * y14 - x14 * y13) < eps)\
            or (abs(x23 * y24 - x24 * y23) < eps)

    def display_img(self, name, img):
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        line_type = cv2.LINE_AA

        # text = f'{self.error_text}{self.properties.detector_choice.name} / {self.properties.descriptor_choice.name}' \
        #        f' / {self.properties.matcher_choice.name} / {self.properties.matcher_method_choice.name}' \
        #        f' / {self.properties.homography_method_choice.name}'
        text = '{}{}/{}/{}/{}/{}'.format(self.error_text, self.properties.detector_choice.name,
                                         self.properties.descriptor_choice.name, self.properties.matcher_choice.name,
                                         self.properties.matcher_method_choice.name,
                                         self.properties.homography_method_choice.name)
        img = cv2.putText(img, text, (0, 15), font, font_scale, (255, 255, 255), thickness, line_type)
        img = cv2.putText(img, text, (0, 15), font, font_scale, (0, 0, 0), thickness, line_type)
        cv2.imshow(name, img)
        k = chr(cv2.waitKey(1) & 255)
        if k == 'a':
            self.properties.update_detector()
            self.update_template()
        elif k == 'z':
            self.properties.update_descriptor()
            self.update_template()
        elif k == 'e':
            self.properties.update_matcher()
        elif k == 'r':
            self.properties.update_matcher_method()
        elif k == 't':
            self.properties.update_homography_method()

    def run(self):
        # load query
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, img = cap.read()  # Capture frame-by-frame
                if not ret:
                    raise KeyboardInterrupt
                # print('Loading query image {}'.format(name))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if self.properties.color \
                    else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print('  Calculating features ...')
                query_kp, query_des = self.calculate_feature_points(img)
                if query_des is None or query_des.size == 0:
                    self.display_img('img', img)
                    continue

                # for each template, calculate the best match
                list_good_matches = []
                for template in self.templates:
                    # print('Estimating match between {} and capture'.format(templ_name))
                    gm = self.find_match(template.des, query_des, template.kp, query_kp)
                    list_good_matches.append(gm if len(gm) >= MIN_MATCH_COUNT else [])
                if not any(list_good_matches):  # if all matches list are empty
                    self.display_img('img', img)
                    continue

                # Get closest template
                best_template = self.template_classif(list_good_matches)

                # Keep the best result
                template = self.templates[best_template]
                good_matches = list_good_matches[best_template]

                src_pts = np.float32([template.kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([query_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                matrix, mask = cv2.findHomography(src_pts, dst_pts, self.properties.homography_method)
                if matrix is None:
                    self.display_img('img', img)
                    continue

                h, w = template.img.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                if not self.is_valid_square(dst):
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img = cv2.polylines(img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                    self.display_img('img', img)
                    continue

                img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                self.display_img('img', img)
                continue

        except KeyboardInterrupt:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Image Classification and Matching Using Local Features and Homography.')
    parser.add_argument('-t', dest='template_names', nargs='+', required=True, help='List of template images')

    args = parser.parse_args()

    # load template images
    identifier = Identifier(args.template_names)
    identifier.run()


if __name__ == "__main__":
    sys.exit(main())
