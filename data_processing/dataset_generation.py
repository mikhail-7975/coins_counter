import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from .annotation_processing import parse_cvat_annotations
from .image_processing import (
    cut_polygon_from_image,
    paste_polygon_on_image,
    rotate_image,
)


class DatasetGenerator:
    def __init__(self):
        self.bg_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.5),
                A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4)),
            ]
        )
        self.object_transforms = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.5),
                A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4)),
            ]
        )

    def generate_dataset(
        self,
        background_images_folder,
        object_images_folder,
        annotation_file,
        object_names,
        result_folder,
    ):
        """
        background_images_folder - path to folder with background images

        object_images - path to folder with objects to paste

        annotation_file - path to file with annotation in CVAT 1.0 format
        """
        annotations = parse_cvat_annotations(annotation_file)
        bg_images = list(Path(background_images_folder).glob("*.jpg"))

        for num, bg_img_path in enumerate(bg_images):
            print("image", bg_img_path)
            bg_img = cv2.imread(bg_img_path)
            if bg_img is None:
                print(f"can not read image {bg_img}")
                continue
            bg_h, bg_w, _ = bg_img.shape
            min_size = min(bg_h, bg_w)
            bg_img = self.bg_transforms(image=bg_img)["image"]
            for money in object_names:
                print("paste", money)
                ann = annotations[money]
                money_number = np.random.randint(len(ann))
                full_money_img = cv2.imread(
                    str(Path(object_images_folder, ann[money_number]["image_path"]))
                )
                full_money_img = self.object_transforms(image=full_money_img)["image"]
                money_img, (money_w, money_h) = cut_polygon_from_image(
                    full_money_img, ann[money_number]["polygon"]
                )
                resize_coef = np.random.randint(5, 8)
                money_dsize = min_size // resize_coef, min_size // resize_coef
                resized_money_image = cv2.resize(money_img, money_dsize)
                rotated = rotate_image(resized_money_image, np.random.randint(0, 360))
                x = np.random.randint(0, bg_w - money_dsize[0])
                y = np.random.randint(0, bg_h - money_dsize[1])
                bg_img = paste_polygon_on_image(bg_img, rotated, (x, y))
            print("save image", num)
            cv2.imwrite(str(Path(result_folder, f"img_{int(time.time())}.jpg")), bg_img)
            time.sleep(1)
