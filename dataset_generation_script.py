import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_cvat_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = defaultdict(list)
    for image in root.findall("image"):
        image_path = image.get("name")
        for polygon in image.findall("polygon"):
            item_name = polygon.get("label")
            points = polygon.get("points")
            points_list = [
                tuple(map(float, point.split(","))) for point in points.split(";")
            ]
            annotation = {
                "item_name": item_name,
                "polygon": points_list,
                "image_path": image_path,
            }
            annotations[annotation["item_name"]].append(annotation)
    return annotations


def cut_polygon_from_image(image, polygon_points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_points, dtype=np.int32)], 255)
    polygon_area = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(np.array(polygon_points, dtype=np.int32))
    cropped_polygon = polygon_area[y : y + h, x : x + w]
    return cropped_polygon, (w, h)


def paste_polygon_on_image(base_image, polygon_image, position):
    x, y, _, _ = position
    h, w, _ = polygon_image.shape
    if x + w > base_image.shape[1] or y + h > base_image.shape[0]:
        raise ValueError(
            "Polygon does not fit within the base image at the given position."
        )
    mask = cv2.cvtColor(polygon_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    result_image = base_image.copy()
    roi = result_image[y : y + h, x : x + w]
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    polygon_fg = cv2.bitwise_and(polygon_image, polygon_image, mask=mask)
    combined = cv2.add(roi_bg, polygon_fg)
    result_image[y : y + h, x : x + w] = combined
    return result_image


xml_file = str(Path(r"C:\Users\mIkhail7975\Desktop\coins\annotations.xml"))
annotations = parse_cvat_annotations(xml_file)


bg_images = list(Path(r"C:\Users\mIkhail7975\Desktop\coins\backgrounds").glob("*.jpg"))
MONEY_KEYS = ["1_ruble", "2_ruble", "5_ruble", "10_ruble", "tail"]
MONEY_IMG_DIR = Path(r"C:\Users\mIkhail7975\Desktop\coins\images")

for bg_img_path in bg_images:
    bg_img = cv2.imread(bg_img_path)
    bg_img.copy()
    bg_h, bg_w, _ = bg_img.shape
    min_size = min(bg_h, bg_w)
    for money in MONEY_KEYS:
        ann = annotations[money]
        money_number = np.random.randint(len(ann))
        full_money_img = cv2.imread(
            str(Path(MONEY_IMG_DIR, ann[money_number]["image_path"]))
        )
        money_img, (money_w, money_h) = cut_polygon_from_image(
            full_money_img, ann[money_number]["polygon"]
        )
        resize_coef = np.random.randint(3, 5)
        money_dsize = money_w // resize_coef, money_h // resize_coef
        resized_money_image = cv2.resize(money_img, money_dsize)
        x = np.random.randint(0, bg_w - money_dsize[0])
        y = np.random.randint(0, bg_h - money_dsize[1])
        bg_img = paste_polygon_on_image(
            bg_img, money_img, (x, y, money_dsize[0], money_dsize[1])
        )
    plt.imshow(bg_img)
    plt.show()
