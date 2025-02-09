import xml.etree.ElementTree as ET
from collections import defaultdict


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
