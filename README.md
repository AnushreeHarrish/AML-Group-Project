# Industrial Waste Classification using Deep Learning

**Course:** EEEM068 Applied Machine Learning  
**Group:** 2



## Overview

Sorting recyclable waste accurately is one of those problems that sounds straightforward but gets complicated fast once you are dealing with 28 visually similar categories under real industrial conditions. This project tackles that challenge by fine-tuning a Vision Transformer (ViT-B/16) on the WaRP-C dataset - a collection of cropped waste images captured from actual recycling conveyor belts.

The core problems we had to deal with were class imbalance (some waste types appear far more often than others) and visual similarity between sub-classes (the difference between a transparent bottle and an opaque one is subtle). To handle these, we built a training pipeline around Focal Loss, weighted random sampling, layer-wise learning rate decay (LLRD), and augmentation strategies including RandAugment, MixUp, and CutMix - all with hyperparameters tuned via Optuna.

Our best model reached **77.11% test accuracy**, a **macro F1 of 0.7678**, and an **AUC of 0.9808** on the 28-class test set. We also ran a CLIP zero-shot baseline for comparison (it scored 0.52%, showing just how much task-specific fine-tuning matters here), and applied INT8 quantisation, which cut the model size by 74% with no meaningful accuracy drop.

Beyond classification, the project also covers object detection on WaRP-D using YOLOv8L and semantic segmentation on WaRP-S using DeepLabV3+ with an EfficientNet-B3 backbone - giving a complete picture of what a full automated waste recognition pipeline looks like.



## Dataset

We used the **WaRP (Waste Recognition and Processing)** dataset, which has three subsets:

|Subset|Task|Details|
|-|-|-|
|WaRP-C|Classification|7,058 train / 1,765 val / 1,551 test images across 28 classes|
|WaRP-D|Object Detection|2,452 train / 522 test images with YOLO-format bounding boxes|
|WaRP-S|Segmentation|112 images with pixel-level masks across 39 classes|

**Dataset Links:**

* WaRP-C-preprocessed: [https://drive.google.com/file/d/1Lsg-UFfYbeHGYx1pz9MzqUlstsNHI4VL/view?usp=sharing](https://drive.google.com/file/d/1Lsg-UFfYbeHGYx1pz9MzqUlstsNHI4VL/view?usp=sharing)
* WaRP-C: [https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset)
* WaRP-D: [https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset)
* WaRP-S: [https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset)



## Models Trained

We trained and compared five architectures on WaRP-C:

|Model|Role|
|-|-|
|**ViT-B/16**|Main model - Vision Transformer with ImageNet-21k pre-trained weights|
|**ConvNeXt-Tiny**|Modernised CNN with ViT-inspired design choices|
|**CoAtNet-0**|Hybrid combining depthwise convolutions and self-attention|
|**Swin Transformer**|Hierarchical ViT variant with shifted window attention|
|**EfficientNet-B0**|Compact CNN with compound scaling|

For detection we used **YOLOv8L** (pre-trained on COCO), and for segmentation **DeepLabV3+** with an EfficientNet-B3 encoder (pre-trained on ImageNet).



## Results

### Classification - Multi-Model Comparison on WaRP-C Test Set

|Model|Accuracy|Precision|Recall|F1|AUC|mAP|
|-|-|-|-|-|-|-|
|**ViT-B/16**|**80.34%**|79.90%|80.31%|**79.56%**|**98.87%**|**86.19%**|
|ConvNeXt-Tiny|78.01%|76.34%|80.98%|77.40%|98.00%|—|
|CoAtNet-0|75.11%|76.62%|75.11%|74.86%|97.84%|—|
|Swin-Base|72.15%|71.30%|78.70%|73.10%|97.97%|81.02%|
|EfficientNet-B0|70.73%|70.46%|70.73%|70.17%|96.50%|—|

### ViT-B/16 Final Test Metrics

|Metric|Value|
|-|-|
|Accuracy|77.11%|
|Macro Precision|0.7523|
|Macro Recall|0.8086|
|Macro F1|0.7678|
|Macro AUC|0.9808|
|mAP|0.8491|

### ViT vs CLIP Zero-Shot

|Metric|ViT-B/16|CLIP (zero-shot)|
|-|-|-|
|Accuracy|77.11%|0.52%|
|Macro F1|0.7678|0.0012|

### Quantisation Results

|Metric|FP32|INT8|
|-|-|-|
|Model Size|327.4 MB|84.4 MB (–74.2%)|
|Accuracy|85.00%|86.88%|
|Speed|356.83 ms/img|304.82 ms/img|

### Detection (WaRP-D) - YOLOv8L

|Model|Precision|Recall|mAP50|mAP50-95|
|-|-|-|-|-|
|YOLOv8L|62.1%|54.5%|59.2%|46.2%|
|YOLOv8 \[prior work]|44.2%|51.3%|54.6%|—|
|YOLOv5 \[prior work]|47.8%|56.9%|40.6%|—|

### Segmentation (WaRP-S) - DeepLabV3+

|Model|Train Loss|Val Loss|Best mIoU|Epochs|
|-|-|-|-|-|
|DeepLabV3+ |0.4190|0.5920|0.6705|30|



## Project Structure

AML-Group-Project/
│
├── NOTEBOOKS/
│   ├── AML\_WarpC\_Processing\_28Classes.ipynb   # shared preprocessing pipeline
│   ├── ViT\_WarpC.ipynb                        # ViT-B/16 — best model
│   ├── vit\_model.ipynb                        # ViT base version
│   ├── convnext\_model.ipynb                   # ConvNeXt-Tiny
│   ├── CoAtNetModel\_with\_outputs.ipynb        # CoAtNet-0
│   ├── swin\_base\_model.ipynb                  # Swin-Base
│   ├── EfficientNet\_model.ipynb               # EfficientNet-B0
│   ├── WARP\_S\_Segmentation.ipynb              # WaRP-S DeepLabV3+
│   ├── Warp\_D\_YOLO(L).ipynb                   # WaRP-D YOLOv8
│   ├── model\_comparision.ipynb                # 5-model comparison
│   └── FILE.ipynb                             # utility notebook
│
├── PROJECT\_REPORT/
│   └── report.pdf                             # 5-page IEEE format report
│
├── REPORT\_IMAGES/
│   └── Report\_Images.pdf                      # all figures used in report
│
├── RESEARCH\_PAPERS/
│   ├── ViT\_Research.pdf                       # Dosovitskiy et al. 2021
│   ├── CoAtNet Research Paper.pdf             # Dai et al. 2021
│   ├── ConvNext\_research.pdf                  # Liu et al. 2022
│   ├── Swin\_Transformer\_paper.pdf             # Liu et al. 2021
│   ├── WaRP.pdf                               # Yudin et al. 2024
│   ├── paper aml (yolo).pdf                   # YOLOv8 reference
│   ├── PAPER-1.pdf                            # additional reference
│   └── PAPER-2.pdf                            # additional reference
│
├── Kaggle\_Dataset\_Link                        # link to WaRP dataset on Kaggle
└── README.md



### Running on Google Colab



WaRP-C:

1. Upload your WaRP-C preprocessed zip to Google Drive
2. Open any notebook from 'NOTEBOOKS/' in Colab
3. Mount Drive and update the dataset path in the config cell
4. Run all cells top to bottom - training loop takes \~2-4 hours on T4 GPU



WaRP-S:

1. Open NOTEBOOKS/WARP\_S\_Segmentation.ipynb in Google Colab
2. Cell 0 downloads the WaRP-S dataset from Kaggle automatically (skip this if you already have it saved to Drive)
3. Update the IMG\_DIR and MASK\_DIR paths in Cell 2 to point to your Drive location, then run all cells in order
4. Training takes approximately 15 minutes on a T4 GPU



WaRP-D:

1. Upload WaRP-D dataset to Google Drive
2. Open a notebook from 'NOTEBOOKS/' in Colab
3. Mount Drive and update the dataset path in the config cell
4. Run all cells top to bottom - training loop takes \~30mins-1 hour on T4 GPU



## Dependencies

The main libraries used across all three tasks:

* pip install timm torch torchvision scikit-learn seaborn matplotlib
* pip install segmentation-models-pytorch albumentations  # for WaRP-S
* pip install ultralytics                                  # for WaRP-D
* pip install transformers                                 # for CLIP
* pip install optuna                                       # for hyperparameter tuning
* Python 3.x
* PyTorch
* timm (for ViT, ConvNeXt, Swin, CoAtNet, EfficientNet)
* ultralytics (for YOLOv8)
* segmentation-models-pytorch (for DeepLabV3+)
* Optuna (hyperparameter search)
* albumentations (image augmentation, especially for segmentation)
* torchvision
* numpy, pandas, matplotlib, scikit-learn



## Team Contributions

|Team Member|Contributions|
|-|-|
|**Surabhi Shanbhogh**|Dataset Overview, Literature Review, Data Preprocessing, Worked on ConvNeXt Model, Model Comparison, Contributed equally to WaRP-C, WaRP-D, WaRP-S and Report Writing|
|**Anushree Harrish**|Dataset Overview, Literature Review, Data Preprocessing, Worked on CoAtNet Model, Model Comparison, Contributed equally to WaRP-C, WaRP-D, WaRP-S and Report Writing|
|**Nikitha Erappa Kattimani**|Dataset Overview, Literature Review, Data Preprocessing, Worked on EfficientNet Model, Model Comparison, Contributed equally to WaRP-C, WaRP-D, WaRP-S and Report Writing|
|**Saivignesh Arra**|Dataset Overview, Literature Review, Data Preprocessing, Worked on ViT Model, Model Comparison, Contributed equally to WaRP-C, WaRP-D, WaRP-S and Report Writing|
|**Pushkar Vijay Jadhav**|Dataset Overview, Literature Review, Data Preprocessing, Worked on Swin Model, Model Comparison, Contributed equally to WaRP-C, WaRP-D, WaRP-S and Report Writing|





*EEEM068 Applied Machine Learning - Group 2*

