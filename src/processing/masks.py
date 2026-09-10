from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def polygon_to_mask(
    segmentation: list,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Convert a polygon segmentation into a binary mask.

    Parameters
    ----------
    segmentation:
        Polygon coordinates in the format
        [x1, y1, x2, y2, ...].

    image_size:
        Image size as (width, height).

    Returns
    -------
    np.ndarray
        Binary mask with shape (height, width).
    """

    width, height = image_size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    polygon = np.array(segmentation).reshape(-1, 2)

    draw.polygon(
        [tuple(point) for point in polygon],
        fill=1,
    )

    return np.array(mask, dtype=np.uint8)


def annotations_to_masks(
    annotations: list,
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """
    Convert all polygon annotations from an image into binary masks.

    Parameters
    ----------
    annotations:
        List of annotations belonging to the same image.

    image_size:
        Image size as (width, height).

    Returns
    -------
    list[np.ndarray]
        One binary mask per annotation.
    """

    masks = []

    for annotation in annotations:
        segmentation = annotation["segmentation"][0]

        mask = polygon_to_mask(
            segmentation=segmentation,
            image_size=image_size,
        )

        masks.append(mask)

    return masks


def build_masks(
    annotations: dict,
) -> dict[str, list[np.ndarray]]:
    """
    Generate binary masks for all images in the dataset.

    Parameters
    ----------
    annotations:
        Loaded annotation dictionary.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Mapping from image filename to a list of binary masks.
    """

    annotations_by_image = {}

    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]

        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(annotation)

    masks_by_filename = {}

    for image in annotations["images"]:
        image_id = image["id"]
        file_name = image["file_name"]

        image_annotations = annotations_by_image.get(image_id, [])

        image_size = (
            image["width"],
            image["height"],
        )

        masks_by_filename[file_name] = annotations_to_masks(
            annotations=image_annotations,
            image_size=image_size,
        )

    return masks_by_filename