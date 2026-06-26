## 1. `decomposition_net_train.py`

* Trains the **decomposition network** (Stage 1).
* Decomposes low-light images into two components:

  * **Reflectance (R):** the intrinsic color and texture of objects.
  * **Illumination (I):** the lighting conditions.
* Uses paired low-light/normal-light images from the LOL dataset.
* Key assumption: paired images share the same reflectance.
* Does not require ground-truth decomposition maps.

## 2. `illumination_adjustment_net_train.py`

* Trains the **illumination adjustment network** (Stage 2).
* Learns to enhance the illumination map.
* Establishes a mapping from low-light illumination to normal-light illumination.
* Computes the ratio between normal-light and low-light illumination maps.
* Enables flexible user-controlled illumination adjustment.

## 3. `reflectance_restoration_net_train.py`

* Trains the **reflectance restoration network** (Stage 3).
* Denoises and restores the reflectance component.
* Incorporates the **Multi-Scale Illumination Attention (MSIA)** module.
* Reduces visual artifacts such as non-uniform blotching and over-smoothing.
* The MSIA module is the key innovation of KinD++ over the original KinD.

## 4. `model.py`

* Contains the neural network architectures for all three subnetworks.
* Defines the network structures (layers and connections).
* Implements the decomposition, illumination adjustment, and restoration models.
* Serves as the core building blocks of the entire framework.

## 5. `msia_BN_3_M.py`

* Implements the **Multi-Scale Illumination Attention (MSIA)** module.
* Introduces the main improvement that distinguishes KinD++ from KinD.
* Employs multi-scale attention mechanisms.
* Contributes to:

  * Better detail preservation.
  * Reduced visual artifacts.
  * Adaptive feature enhancement.

## 6. `evaluate.py`

* Evaluates the trained model on custom test images.
* Loads test images from the `test_images` directory.
* Executes the complete enhancement pipeline:

  1. Decomposition → 2. Illumination Adjustment → 3. Reflectance Restoration.
* Saves the enhanced output images.

## 7. `evaluate_LOLdataset.py`

* Evaluates model performance specifically on the LOL dataset.
* Computes image quality metrics (e.g., PSNR, SSIM).
* Compares enhanced images with normal-light ground-truth images.
* Used for quantitative benchmarking.

## 8. `utils.py`

* Contains utility functions used throughout the project, including:

  * Image loading and saving.
  * Data preprocessing.
  * Loss function implementations.
  * Evaluation metric computation.
  * Common operations (e.g., gradient computation and smoothness constraints).

## Training Pipeline

The three subnetworks are trained sequentially:

```bash
python decomposition_net_train.py          # Stage 1
python illumination_adjustment_net_train.py  # Stage 2
python reflectance_restoration_net_train.py  # Stage 3
```

## Evaluation

```bash
python evaluate.py             # Evaluate on custom images
python evaluate_LOLdataset.py  # Evaluate on the LOL dataset
```

## Key Innovation

The primary contribution of KinD++ is the **MSIA module**, integrated into the reflectance restoration network. By leveraging multi-scale attention, MSIA provides more effective noise suppression while preserving fine image details, resulting in superior enhancement quality compared with the original KinD framework.
