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

The project already has a solid data pipeline and a working segmentation workflow, and it has been further improved with recent experiments focused on model optimization and generalization:

- loading and cleaning of dataset annotations,
- conversion of polygon annotations into binary masks,
- removal of duplicate annotations,
- construction of masks per image file,
- division of the dataset into train/validation/test splits,
- definition of a PyTorch dataset for images and masks,
- use of a U-Net architecture as the main segmentation baseline,
- training with segmentation losses and evaluation metrics,
- support for DataLoaders, checkpointing, and metric monitoring,
- hyperparameter optimization via random search,
- use of data augmentation to improve robustness and reduce overfitting,
- validation against Kaggle submission metrics.

### Main project structure

- `src/datasets/segmentation.py`: dataset for images and masks.
- `src/processing/masks.py`: generation of masks from polygon annotations.
- `src/splitting/split.py`: dataset partitioning.
- `src/models/unet.py`: U-Net architecture used for segmentation.
- `src/training/losses.py`: loss functions such as Dice and BCE + Dice.
- `src/training/metrics.py`: evaluation metrics such as Dice and IoU.
- `src/training/trainer.py`: training and validation loop.
- `src/utils/data.py`: utilities for loading annotations and handling data.
- `config/config.yaml`: general project configuration.

## Recent improvements

The latest experiments focused on the two main factors that had the greatest impact on performance:

- random search over the most relevant hyperparameters of the network,
- addition of data augmentation during training to improve generalization.

This led to a clear improvement in the Kaggle submission score:

- previous submission score: 0.25
- improved submission score: 0.29

This represents a meaningful gain and indicates that the model is moving in the right direction, especially when the optimization is tied to more robust training data transformations.

## Current status of the project

The repository is no longer just a baseline prototype. It has moved to a more mature experimental stage, with a functional pipeline and a clear path toward performance improvements.

At the moment, the project already includes:

- robust data preparation,
- a PyTorch segmentation pipeline,
- a trained U-Net baseline,
- standard loss functions and evaluation metrics,
- hyperparameter optimization through random search,
- data augmentation strategies,
- validated performance gains in Kaggle submissions.

This means the project already has a strong foundation for iterative improvement and is now in a phase where model tuning and training strategy have a measurable impact on results.

## What still needs to be improved

Although the recent results are promising, there is still room to continue improving the solution:

- further hyperparameter tuning,
- testing alternative normalization strategies,
- more aggressive and task-specific augmentation,
- exploration of deeper or more advanced segmentation architectures,
- improving the balance between precision and recall,
- postprocessing of masks to remove noise or isolated components,
- comparing additional loss functions,
- stronger validation with challenge-relevant metrics.

## Strategies that likely need to be tested next

The current U-Net is already proving useful, but there are still several directions worth exploring to push performance further:

- deeper U-Net variants or improved encoder-decoder designs,
- UNet++ or models with stronger skip connections,
- DeepLabV3+ for broader contextual understanding,
- FPN and other semantic segmentation architectures,
- pretrained backbones for better feature extraction,
- domain-specific augmentation tailored to solar imagery,
- thresholding and morphological refinement of final masks,
- ensemble strategies to improve stability.

## Conclusion

This project has evolved from a basic segmentation pipeline into a more structured and optimized workflow. The main breakthrough in the latest stage was the combination of random search for hyperparameter tuning and data augmentation, which improved the Kaggle submission score from 0.25 to 0.29.

The current goal is to continue refining the training setup, explore more advanced architectures, and strengthen the final segmentation quality through careful validation and targeted optimization.

> In short: the pipeline is now functioning at a stronger level, and the latest results show that systematic tuning and augmentation are producing measurable improvements in performance.
