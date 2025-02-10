from pathlib import Path

from data_processing.yolo2classification import Yolo2ClassificationConverter
from data_processing.yolo_dataset_generation import YoloDatasetGenerator

g = YoloDatasetGenerator(n_images=150)

c = Yolo2ClassificationConverter()

if __name__ == "__main__":
    bg_imgs_folder = Path(r"C:\Users\mIkhail7975\Desktop\coins\backgrounds")
    object_images_folder = Path(r"C:\Users\mIkhail7975\Desktop\coins\images")
    annotation_path = Path(r"C:\Users\mIkhail7975\Desktop\coins\annotations.xml")
    MONEY_KEYS = ["1_ruble", "2_ruble", "5_ruble", "10_ruble", "tail"]
    dst_folder = Path(r"C:\Users\mIkhail7975\Desktop\coins\generated_dataset_2")

    dst_imgs_folder = Path(dst_folder, "images")
    dst_imgs_folder.mkdir(exist_ok=True, parents=True)
    dst_labels_folder = Path(dst_folder, "labels")
    dst_labels_folder.mkdir(exist_ok=True, parents=True)

    # g.generate_dataset(
    #     bg_imgs_folder, object_images_folder, annotation_path, MONEY_KEYS, dst_folder
    # )

    c.convet(
        Path(dst_folder, "images"),
        Path(dst_folder, "labels"),
        Path(r"C:\Users\mIkhail7975\Desktop\coins\classification"),
    )
