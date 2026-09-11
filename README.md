# Kaggle Solar Filament Segmentation

This project is part of a solar image segmentation challenge where the goal is to detect and segment solar filaments from images of the Sun. The task is not simply to classify whether a filament is present or not, but to generate a binary mask that accurately captures the shape and extent of each filament while keeping false positives as low as possible.

The final objective is to build a deep learning pipeline that produces useful masks for each image, with a clear separation between filament regions and the solar background.

## Problem context

In this kind of imagery, filaments appear as relatively thin structures with irregular shapes on the solar surface. The segmentation task is challenging for several reasons:

- filaments can have very different widths,
- they present complex boundaries and fine details,
- the solar background may include textures and artifacts that make separation harder,
- the target mask is usually small compared with the total image size, which creates a class imbalance problem.

For that reason, the solution depends not only on the model architecture, but also on proper preprocessing, a solid training strategy, and careful validation.

## What has been done so far

The project already has a functional foundation for preparing data and training a segmentation model:

- loading and cleaning of dataset annotations,
- conversion of polygon annotations into binary masks,
- removal of duplicate annotations,
- construction of masks per image file,
- division of the dataset into train/validation/test splits,
- definition of a PyTorch dataset for images and masks,
- use of a small U-Net as a segmentation baseline,
- training with loss functions and evaluation metrics,
- support for DataLoaders, checkpointing, and metric monitoring.

### Main project structure

- `src/datasets/segmentation.py`: dataset for images and masks.
- `src/processing/masks.py`: generation of masks from polygon annotations.
- `src/splitting/split.py`: dataset partitioning.
- `src/models/unet.py`: base U-Net architecture.
- `src/training/losses.py`: loss functions such as Dice and BCE + Dice.
- `src/training/metrics.py`: evaluation metrics such as Dice and IoU.
- `src/training/trainer.py`: training and validation loop.
- `src/utils/data.py`: utilities for loading annotations and handling data.
- `config/config.yaml`: general project configuration.

## Current status of the project

The repository is in an early stage, with a very useful base for experimentation, but it cannot yet be considered an optimized final solution.

At the moment, the core infrastructure has been set up:

- data preparation,
- a PyTorch segmentation pipeline,
- a simple baseline U-Net model,
- appropriate loss and metrics for binary segmentation.

This means there is already a solid foundation for training and evaluating prototypes, but there is still a lot of work ahead to improve performance, stability, and robustness.

## What still needs to be improved

This project is far from being complete or fully optimized. Several important areas still need to be tested and refined:

- image preprocessing optimization,
- testing different normalization strategies and data augmentation,
- exploration of alternative segmentation architectures,
- fine-tuning of hyperparameters,
- improving the balance between precision and recall,
- postprocessing masks to remove noise or small isolated components,
- comparison of different loss functions,
- more rigorous validation with challenge-relevant metrics.

## Strategies that likely need to be tested

Although the U-Net is a good baseline, other strategies still need to be explored to determine which works best for this type of solar imagery:

- deeper U-Nets or improved variants,
- UNet++ or models with stronger skip connections,
- DeepLabV3+ to capture broader context,
- FPN and other semantic segmentation architectures,
- models with pretrained backbones,
- domain-specific data augmentation for astronomical images,
- thresholding and morphological refinement of final masks,
- ensembles of multiple models to stabilize predictions.

## Conclusion

This project starts from a reasonable foundation: data cleaning, mask generation, training of a segmentation model, and evaluation with standard metrics. However, there is still a lot to optimize and test before it can be considered a competitive solution.

The current goal is to use this structure as a starting point for rapid iteration, comparing models, adjusting hyperparameters, and exploring more advanced strategies that improve the accuracy of solar filament segmentation.

> In short: the base pipeline is in place, but the most important part is still to come: optimization, deeper validation, and testing of alternative architectures and training strategies.
