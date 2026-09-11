from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label


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


def get_connected_components(
    binary_mask: np.ndarray,
    connectivity: int = 2,
) -> tuple[np.ndarray, int]:
    labels, num_components = label(
        binary_mask,
        connectivity=connectivity,
        return_num=True,
    )

    return labels, num_components


def calculate_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return intersection / union


def filter_components_by_area(
    labeled_mask: np.ndarray,
    min_area: int = 664,
) -> np.ndarray:
    """Remove connected components smaller than a minimum area."""
    filtered_mask = np.zeros_like(labeled_mask)

    for component_id in range(1, labeled_mask.max() + 1):
        component = labeled_mask == component_id

        if component.sum() >= min_area:
            filtered_mask[component] = component_id

    return filtered_mask