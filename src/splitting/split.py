from sklearn.model_selection import train_test_split


def split_masks(
    masks: dict,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[dict, dict, dict]:
    """
    Split image masks into training, validation and test sets.

    The split is performed at image level, so all masks belonging
    to the same image remain in the same subset.

    Parameters
    ----------
    masks:
        Dictionary mapping image filenames to their masks.

    train_size:
        Proportion of images assigned to the training set.

    val_size:
        Proportion of images assigned to the validation set.

    test_size:
        Proportion of images assigned to the test set.

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    tuple[dict, dict, dict]
        Training, validation and test dictionaries.
    """

    if not abs(train_size + val_size + test_size - 1.0) < 1e-8:
        raise ValueError(
            "train_size, val_size and test_size must sum to 1."
        )

    filenames = list(masks.keys())

    train_files, temp_files = train_test_split(
        filenames,
        train_size=train_size,
        random_state=random_state,
    )

    relative_test_size = test_size / (val_size + test_size)

    val_files, test_files = train_test_split(
        temp_files,
        test_size=relative_test_size,
        random_state=random_state,
    )

    train_masks = {
        filename: masks[filename]
        for filename in train_files
    }

    val_masks = {
        filename: masks[filename]
        for filename in val_files
    }

    test_masks = {
        filename: masks[filename]
        for filename in test_files
    }

    return train_masks, val_masks, test_masks