# references used :
# https://github.com/dmartinalbo/image-matching
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
import sys
import argparse
import numpy as np
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
        self.template_cm_size = 4.23  # cm

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
        return cv2.isContourConvex(square_pts)

    def display_img(self, name, img):
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        line_type = cv2.LINE_AA

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

    def compute_dist(self, img, target_pts, tvec):
        side_length = np.mean([np.linalg.norm(target_pts[0].ravel() - target_pts[1].ravel()),
                               np.linalg.norm(target_pts[1].ravel() - target_pts[2].ravel()),
                               np.linalg.norm(target_pts[2].ravel() - target_pts[3].ravel()),
                               np.linalg.norm(target_pts[3].ravel() - target_pts[0].ravel())])
        pixels_per_metric = side_length / self.template_cm_size
        dist = tvec[2][0] / pixels_per_metric
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 3
        thickness = 3
        line_type = cv2.LINE_AA
        height, width = img.shape[:2]
        return cv2.putText(img, 'distance : {}cm'.format(round(dist, 2)), (width-500, height-20),
                          font, font_scale, (255, 0, 125), thickness, line_type)

    def draw_axis(self, img, rvec, tvec, length=100):
        axis_points = np.float32([(0, 0, 0), (length, 0, 0), (0, length, 0), (0, 0, length)])

        image_points, _ = cv2.projectPoints(axis_points, rvec, tvec,
                                            self.properties.camera_matrix,
                                            self.properties.dist_coeffs)

        origin = tuple(image_points[0].ravel())
        cv2.line(img, origin, tuple(image_points[1].ravel()), (0, 0, 255), thickness=3)
        cv2.line(img, origin, tuple(image_points[2].ravel()), (0, 255, 0), 3)
        cv2.line(img, origin, tuple(image_points[3].ravel()), (255, 0, 0), 3)

    def run(self):
        # load query
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, img = cap.read()  # Capture frame-by-frame
                if not ret:
                    raise KeyboardInterrupt()
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
                pts = np.expand_dims(np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]), axis=1)
                dst = cv2.perspectiveTransform(pts, matrix)
                if not self.is_valid_square(dst):
                    self.display_img('img', img)
                    continue

                img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    np.insert(pts.reshape(-1, 2), 2, 0, axis=1),
                    dst.reshape(-1, 2),
                    self.properties.camera_matrix,
                    self.properties.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    self.draw_axis(img, rotation_vector, translation_vector)
                    img = self.compute_dist(img, dst, translation_vector)

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
