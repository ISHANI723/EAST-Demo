# Report on EAST: An Efficient and Accurate Scene Text Detector

This repository contains the implementation and demonstration for **SIT789 Robotics, Computer Vision and Speech Processing – High Distinction Task 5.2**.

The project is based on the research paper:

> X. Zhou et al., "EAST: An Efficient and Accurate Scene Text Detector," CVPR 2017.

The aim of this demonstration was to implement the EAST scene text detection pipeline, train it on a customised dataset, test the trained model, and analyse the resulting text-region detections.

## 1. Project Overview

EAST is a scene text detection method that directly predicts text regions from an input image using a fully convolutional network followed by post-processing.

The implementation used in this project uses:

* PyTorch
* VGG16 backbone
* RBOX representation
* Dice loss for the score map
* Adam optimiser
* CPU-based training

The project follows the main EAST pipeline:

```text
Input Image
     ↓
Feature Extraction
     ↓
Multi-level Feature Merging
     ↓
Score Map + Geometry Map
     ↓
Thresholding
     ↓
NMS / Post-processing
     ↓
Detected Text Regions
```

## 2. Dataset

The experiment used a **customised subset of the ICDAR 2015 scene text dataset**.

Instead of using the complete benchmark, a smaller subset was prepared for the demonstration:

| Dataset split | Images |
| ------------- | -----: |
| Training      |    140 |
| Testing       |     30 |
| Total         |    170 |

The corresponding text annotations were used for training and evaluation.

This smaller customised dataset was selected because the task focuses on demonstrating the chosen research method rather than reproducing the complete original benchmark.

## 3. Training Configuration

The EAST model was trained for **20 epochs**.

| Parameter           | Value            |
| ------------------- | ---------------- |
| Training images     | 140              |
| Test images         | 30               |
| Backbone            | VGG16            |
| Representation      | RBOX             |
| Batch size          | 2                |
| Learning rate       | $1\times10^{-4}$ |
| Optimiser           | Adam             |
| Epochs              | 20               |
| Scheduler           | MultiStepLR      |
| Scheduler milestone | Epoch 10         |
| Decay factor        | 0.1              |
| Hardware            | CPU              |
| Final checkpoint    | Epoch 20         |

The final model checkpoint used for testing was:

```text
demo/pths/model_epoch_20.pth
```

## 4. Demonstration

The main testing script is:

```text
test_30.py
```

This script:

1. Loads the trained EAST model.
2. Loads the epoch-20 checkpoint.
3. Reads the 30 test images.
4. Runs EAST detection on each image.
5. Generates visualised detection results.
6. Saves the results to the output directory.

The evaluation script was then used to compare the predicted text regions with the ground-truth annotations.

## 5. Quantitative Results

The final evaluation on the 30 test images produced:

| Metric               | Result |
| -------------------- | -----: |
| Ground-truth regions |    117 |
| Predicted regions    |    714 |
| Ignored predictions  |     23 |
| True positives       |      6 |
| False positives      |    708 |
| False negatives      |    111 |
| Precision            |  0.84% |
| Recall               |  5.13% |
| F1-score             |  1.44% |

The results were substantially lower than those reported in the original EAST paper. This experiment was not intended as a direct reproduction of the original benchmark because the dataset size, training configuration and some implementation components differed.

## 6. Qualitative Results

Several test images were inspected visually to understand the behaviour of the trained detector.

The model was able to generate candidate text regions, but many predictions were incorrectly positioned or occurred in non-text regions. Some detections were close to the ground-truth text, while many others produced false positives.

Representative results include:

* `result_img_343.jpg`
* `result_img_709.jpg`
* `result_img_711.jpg`
* `result_img_857.jpg`
* `result_img_880.jpg`
* `result_img_932.jpg`

These examples were generated from the project's own test results and were not copied from the original EAST paper.

## 7. Training Data Diagnostic

During the experiment, the training samples were also inspected to check whether the generated score maps contained positive text pixels.

The diagnostic showed:

```text
Training samples:       140
Samples with text:       56
Samples with no text:    84

Positive samples:        40.0%
No-positive samples:     60.0%
```

This indicated that a large proportion of the training samples did not contain positive score-map pixels, which was identified as an important limitation of the current training setup.

## 8. Implementation Differences

There were several differences between this demonstration and the original EAST implementation.

### Loss function

The original EAST paper uses a class-balanced cross-entropy loss for the score map.

The PyTorch implementation used in this project uses **Dice loss** for the score map.

### NMS

The original EAST method uses **locality-aware NMS (LANMS)**.

Due to dependency and compilation issues with the original LANMS package in the Windows/Python environment, a Python IoU-based NMS fallback was used for the final experiment.

Therefore, the results should be considered an implementation and demonstration of the EAST pipeline rather than an exact reproduction of the original benchmark.

## 9. Limitations

The main limitations identified during the experiment were:

* Small training dataset.
* Large proportion of training samples without positive score-map pixels.
* Differences between the project implementation and the original EAST loss.
* LANMS dependency issues requiring an alternative NMS implementation.
* CPU-only training, which limited the number of experiments and hyperparameter configurations that could be tested.

## 10. Future Improvements

The following improvements could be implemented in future work:

1. Use a larger and more balanced training dataset.
2. Increase the number of useful positive training samples.
3. Reproduce the original EAST score-map loss more closely.
4. Implement the original locality-aware NMS.
5. Experiment with different training settings and hyperparameters.
6. Evaluate the model on the complete ICDAR 2015 benchmark.
7. Explore curved text detection and integration with text recognition.

## 11. Reference

Zhou, X., Yao, C., Wen, H., Wang, Y., Zhou, S., He, W. and Liang, J. (2017).
**EAST: An Efficient and Accurate Scene Text Detector.**
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5551–5560.

DOI: `10.1109/CVPR.2017.283`

---

**Author:** Ishani Bhongale
**Unit:** SIT789 – Robotics, Computer Vision and Speech Processing
**Assessment:** High Distinction Task 5.2 – Minor Research Project
**Year:** 2026
