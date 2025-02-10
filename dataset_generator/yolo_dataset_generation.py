import random
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


class YoloDatasetGenerator:
    def __init__(
        self,
        n_images,
        relative_object_size_range=(0.125, 0.2),
        object_count_range=(4, 10),
    ):
        self.n_images = n_images
        s2, s1 = relative_object_size_range
        self.object_size_range = (int(1 / s1), int(1 / s2))
        self.object_count_range = object_count_range
        self.bg_transforms = A.Compose(
            [
                A.AdditiveNoise(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0), p=1.0),
                A.Rotate(limit=45, p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.5),
                A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4)),
                A.RandomRain(),
                A.RandomSnow(),
            ]
        )
        self.object_transforms = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.5),
                A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4)),
                A.RandomBrightnessContrast(brightness_limit=(0.0, 0.3), p=0.8),
            ]
        )

    def __bbox2yolo(self, bbox, cls_id, img_w, img_h):
        x, y, w, h = bbox
        x = x + w / 2
        y = y + h / 2
        x, y, w, h = x / img_w, y / img_h, w / img_w, h / img_h
        return f"{cls_id} {x} {y} {w} {h}"

    def __save_results(self, image, annotation, dst_folder):
        out_filename = f"img_{int(time.time_ns())}"
        cv2.imwrite(
            str(Path(dst_folder, "images", f"{out_filename}.jpg")),
            image,
        )
        with open(Path(dst_folder, "labels", f"{out_filename}.txt"), "w") as f:
            for i in annotation:
                f.write(i)
                f.write("\n")
        time.sleep(0.1)

    def __create_dst_dirs(self, dst_folder):
        img_folder = Path(dst_folder, "images")
        img_folder.mkdir(parents=True, exist_ok=True)
        label_folder = Path(dst_folder, "labels")
        label_folder.mkdir(exist_ok=True, parents=True)

    def __generate_object_size(self, bg_width, bg_height):
        """
        generate the size of the inserted object relative to the size
        of the background image

        bg_width - background image width

        bg_width - background image height
        """
        min_size = min(bg_height, bg_width)
        resize_coef = np.random.randint(
            self.object_size_range[0], self.object_size_range[1]
        )
        return min_size // resize_coef, min_size // resize_coef

    def __generate_object_position(self, target_size, object_size):
        bg_w, bg_h = target_size
        obj_w, obj_h = object_size
        x = random.randint(1, bg_w - obj_w - 1)
        y = random.randint(1, bg_h - obj_h - 1)
        return x, y

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
        self.__create_dst_dirs(result_folder)

        annotations = parse_cvat_annotations(annotation_file)
        bg_images = list(Path(background_images_folder).glob("*.jpg"))

        saved_imgs = 0
        while saved_imgs < self.n_images:
            for num, bg_img_path in enumerate(bg_images):

                bg_img = cv2.imread(bg_img_path)
                if bg_img is None:
                    print(f"can not read image {bg_img}")
                    continue

                bg_img = self.bg_transforms(image=bg_img)["image"]

                bg_h, bg_w, _ = bg_img.shape

                objects_markup_list = []
                n_objects = random.randint(
                    self.object_count_range[0], self.object_count_range[1]
                )
                for _ in range(n_objects):
                    # for cls_id, money in enumerate(object_names):
                    cls_id = random.randint(0, len(object_names) - 1)
                    object_name = object_names[cls_id]
                    ann = annotations[object_name]

                    object_sample_number = np.random.randint(len(ann))
                    full_money_img = cv2.imread(
                        str(
                            Path(
                                object_images_folder,
                                ann[object_sample_number]["image_path"],
                            )
                        )
                    )
                    full_money_img = self.object_transforms(image=full_money_img)[
                        "image"
                    ]
                    object_img, (money_w, money_h) = cut_polygon_from_image(
                        full_money_img, ann[object_sample_number]["polygon"]
                    )
                    object_dsize = self.__generate_object_size(bg_w, bg_h)
                    resized_money_image = cv2.resize(object_img, object_dsize)

                    rotated = rotate_image(
                        resized_money_image, np.random.randint(0, 360)
                    )

                    obj_position = self.__generate_object_position(
                        (bg_w, bg_h), object_dsize
                    )
                    bg_img = paste_polygon_on_image(bg_img, rotated, obj_position)

                    money_bbox = *obj_position, *object_dsize
                    objects_markup_list.append(
                        self.__bbox2yolo(money_bbox, cls_id, bg_w, bg_h)
                    )
                self.__save_results(bg_img, objects_markup_list, Path(result_folder))
                saved_imgs += 1
