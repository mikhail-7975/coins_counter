import cv2
import numpy as np


def cut_polygon_from_image(image, polygon_points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_points, dtype=np.int32)], 255)
    polygon_area = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(np.array(polygon_points, dtype=np.int32))
    cropped_polygon = polygon_area[y : y + h, x : x + w]
    return cropped_polygon, (w, h)


def paste_polygon_on_image(base_image, polygon_image, position):
    (
        x,
        y,
    ) = position
    h, w, _ = polygon_image.shape
    assert (
        x + w < base_image.shape[1] and y + h < base_image.shape[0]
    ), "assertion failed"
    mask = cv2.cvtColor(polygon_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    result_image = base_image.copy()
    roi = result_image[y : y + h, x : x + w]
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    polygon_fg = cv2.bitwise_and(polygon_image, polygon_image, mask=mask)
    combined = cv2.add(roi_bg, polygon_fg)
    result_image[y : y + h, x : x + w] = combined
    return result_image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
