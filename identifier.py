import sys
import argparse
import numpy as np
import cv2
import yaml

MIN_MATCH_COUNT = 10


class Template:
    def __init__(self, name, gray_img, color_img, kp, des, grayscale=True):
        self.name = name
        self.img_gray = gray_img
        self.img_color = color_img
        self.img = gray_img if grayscale else color_img
        self.kp = kp
        self.des = des


class Identifier:
    # define the lower and upper boundaries of the colors in the HSV color space
    hsv_blue_ranges = [((90, 75, 75), (135, 255, 175))]
    hsv_green_ranges = [((35, 75, 75), (75, 255, 175))]
    hsv_red_ranges = [((0, 75, 75), (15, 255, 175)), ((165, 75, 75), (179, 255, 175))]
    hsv_black_ranges = [((0, 0, 0), (179, 255, 75))]

    def __init__(self, size_cm):
        self.error_text = ''
        self.template_cm_size = size_cm
        with open('calibration.yaml') as f:
            loadeddict = yaml.load(f)
            self.camera_matrix = np.asarray(loadeddict.get('cameraMatrix'))
            self.dist_coeffs = np.asarray(loadeddict.get('distCoeffs'))

    @staticmethod
    def is_valid_square(pts):
        return cv2.isContourConvex(pts)

    def display_img(self, name, img):
        cv2.imshow(name, img)
        self.process_keys()

    def process_keys(self):
        cv2.waitKey(1)

    # @staticmethod
    # def compute_dist(img, tvec):
    #     dist = np.linalg.norm(tvec)
    #     font = cv2.FONT_HERSHEY_PLAIN
    #     font_scale = 3
    #     thickness = 3
    #     line_type = cv2.LINE_AA
    #     height, width = img.shape[:2]
    #     return cv2.putText(img, 'distance : {}cm'.format(round(dist, 2)), (width - 500, height - 20),
    #                        font, font_scale, (255, 0, 125), thickness, line_type)

    def draw_axis(self, img, rvec, tvec, length=10):
        axis_points = np.float32([(0, 0, 0), (length, 0, 0), (0, length, 0), (0, 0, length)])

        image_points, _ = cv2.projectPoints(axis_points, rvec, tvec,
                                            self.camera_matrix,
                                            self.dist_coeffs)

        origin = tuple(image_points[0].ravel().astype(np.int32, casting='unsafe'))
        point = image_points[1].ravel().astype(np.int32, casting='unsafe')
        cv2.line(img, origin, tuple(point), (0, 0, 255), thickness=3)
        point = image_points[2].ravel().astype(np.int32, casting='unsafe')
        cv2.line(img, origin, tuple(point), (0, 255, 0), thickness=3)
        point = image_points[3].ravel().astype(np.int32, casting='unsafe')
        cv2.line(img, origin, tuple(point), (255, 0, 0), thickness=3)

    @staticmethod
    def crop_min_area_rect(img, rect):
        # rotate img
        angle = rect[2]
        rows, cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(img, M, (cols, rows))

        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv2.boxPoints(rect)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

        return img_crop

    @staticmethod
    def get_contours(img):
        tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), 0)
        tmp_img = cv2.Canny(tmp_img, 35, 135)
        contours = cv2.findContours(tmp_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
        return (elt for elt in contours if elt.shape[0] > 5)

    @staticmethod
    def is_ellipse(contour):
        ellipse = cv2.fitEllipse(contour)

        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                                int(ellipse[2]), 0, 360, 5)
        return cv2.matchShapes(contour, poly, cv2.CONTOURS_MATCH_I1, 0.0) < 0.01

    @staticmethod
    def get_percentage(values_list, ranges):
        is_detected = np.zeros(values_list.shape[:-1])
        for r in ranges:
            tmp = cv2.inRange(values_list, r[0], r[1])
            tmp2 = np.asarray(tmp)
            is_detected = np.add(is_detected, np.asarray(cv2.inRange(values_list, r[0], r[1])))
        return (cv2.countNonZero(is_detected) / is_detected.size) * 100

    @staticmethod
    def draw_detected_form(img, elt, black_percent, blue_percent, red_percent, green_percent):
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        line_type = cv2.LINE_AA

        box = cv2.boxPoints(elt)
        elt = np.int0(box)

        img = cv2.putText(img, '{}%'.format(black_percent), tuple(elt[0]), font, font_scale, (255, 255, 255), thickness,
                          line_type)
        img = cv2.putText(img, '{}%'.format(blue_percent), tuple(elt[1]), font, font_scale, (255, 0, 0), thickness,
                          line_type)
        img = cv2.putText(img, '{}%'.format(red_percent), tuple(elt[2]), font, font_scale, (0, 0, 255), thickness,
                          line_type)
        img = cv2.putText(img, '{}%'.format(green_percent), tuple(elt[3]), font, font_scale, (0, 255, 0), thickness,
                          line_type)

        cv2.drawContours(img, [elt], -1, (0, 0, 0), 2)
        return box

    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, img = cap.read()  # Capture frame-by-frame
                if not ret:
                    raise KeyboardInterrupt()

                for elt in self.get_contours(img):
                    if not self.is_ellipse(elt):
                        continue

                    elt = cv2.minAreaRect(elt)

                    img_croped = self.crop_min_area_rect(img, elt)
                    if img_croped.size == 0:
                        continue
                    hsv_img = cv2.cvtColor(img_croped, cv2.COLOR_BGR2HSV)
                    if hsv_img is None:
                        continue
                    hues = hsv_img[:, :, 0].ravel()

                    black_percent = self.get_percentage(hsv_img, self.hsv_black_ranges)
                    if black_percent < 1 or black_percent > 30:
                        continue
                    blue_percent = self.get_percentage(hsv_img, self.hsv_blue_ranges)
                    red_percent = self.get_percentage(hsv_img, self.hsv_red_ranges)
                    green_percent = self.get_percentage(hsv_img, self.hsv_green_ranges)
                    if blue_percent < 50 and ((red_percent < 50 and green_percent < 50) or black_percent < 7):
                        continue

                    box = self.draw_detected_form(img, elt, black_percent, blue_percent, red_percent, green_percent)

                    real_world_pts = np.float32([[0, 0],
                                                 [0, self.template_cm_size],
                                                 [self.template_cm_size, self.template_cm_size],
                                                 [self.template_cm_size, 0]])

                    (success, rotation_vector, translation_vector) = cv2.solvePnP(
                        np.insert(real_world_pts, 2, 0, axis=1),
                        box,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    if success:
                        self.draw_axis(img, rotation_vector, translation_vector, self.template_cm_size)
                        # img = self.compute_dist(img, translation_vector)

                self.display_img('img', img)
                continue

        except KeyboardInterrupt:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Round colored forms detection')
    parser.add_argument('-s', dest='size_cm', type=float, required=True,
                        help='Real size in centimeter of template images')

    args = parser.parse_args()

    # load template images
    identifier = Identifier(**vars(args))
    identifier.run()


if __name__ == "__main__":
    sys.exit(main())
