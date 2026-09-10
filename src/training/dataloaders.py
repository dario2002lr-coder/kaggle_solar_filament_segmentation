from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.

    Parameters
    ----------
    dataset:
        PyTorch Dataset to load.

    batch_size:
        Number of samples per batch.

    shuffle:
        Whether to shuffle the samples.

    num_workers:
        Number of worker processes used to load data.

    pin_memory:
        Whether to use pinned memory for faster CPU-to-GPU transfers.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation and test datasets.

    The training DataLoader is shuffled, while validation and test
    DataLoaders preserve the dataset order.

    Parameters
    ----------
    train_dataset:
        Training dataset.

    val_dataset:
        Validation dataset.

    test_dataset:
        Test dataset.

    batch_size:
        Number of samples per batch.

    num_workers:
        Number of worker processes used to load data.

    pin_memory:
        Whether to use pinned memory for faster CPU-to-GPU transfers.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Training, validation and test DataLoaders.
    """
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader