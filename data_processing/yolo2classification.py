import time
from pathlib import Path

import cv2


class Yolo2ClassificationConverter:
    def __init__(self, transform=None):
        self.transform = None

    def convet(self, image_dir, annotation_dir, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        annotation_files_list = list(Path(annotation_dir).glob("*.txt"))
        for annotation_file in annotation_files_list:
            image_file_name = annotation_file.name.replace(".txt", ".jpg")
            image_path = Path(image_dir, image_file_name)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {image_file_name} not found.")
                continue

            height, width, _ = image.shape

            # Read the annotation file
            with open(annotation_file, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = parts[0]
                    center_x = float(parts[1]) * width
                    center_y = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height

                    # Calculate the top-left corner of the bounding box
                    x_min = int(center_x - bbox_width / 2)
                    y_min = int(center_y - bbox_height / 2)
                    x_max = int(center_x + bbox_width / 2)
                    y_max = int(center_y + bbox_height / 2)

                    # Crop the bounding box from the image
                    cropped_image = image[y_min:y_max, x_min:x_max]

                    # Save the cropped image with the class ID as part of the filename
                    output_filename = Path(
                        output_dir, f"img_{class_id}_{time.time_ns()}.jpg"
                    )
                    cv2.imwrite(str(output_filename), cropped_image)
