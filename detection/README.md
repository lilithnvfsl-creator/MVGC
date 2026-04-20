# Dynamic Multi-scale Vision Graph Convolution for Efficient Edge Perception in Autonomous Driving (MVGC)


> **🔔 Important Notice:** > This code repository is directly related to our manuscript currently under review at ***The Visual Computer***. If you find our work, code, or data helpful in your research, please consider citing our manuscript (see [Citation](#citation)).

---

## 📖 Abstract

Efficient visual perception is critical for safe autonomous driving on resource-constrained edge devices. Existing lightweight models struggle to balance global context and computational cost, as CNNs lack long-range dependencies while ViTs and graph neural networks suffer from high latency. This work presents **Multi-scale Vision Graph Convolution (MVGC)**, a hybrid architecture that eliminates KNN bottlenecks via dynamic sparse multi-scale graph attention and enhances local features with conditional position encoding. 

Experiments on MS COCO and BDD100K datasets show that MVGC improves object detection AP by 2.98% and instance segmentation AP by 11.51% with state-of-the-art accuracy-latency trade-off, making it suitable for real-time vehicular edge computing.

## 🚀 Highlights & Key Components

* **MSVGA (Multi-scale Vision Graph Attention):** Eliminates the computationally expensive KNN bottleneck found in traditional ViGs by utilizing a dynamic, sparse multi-scale connection strategy with linear complexity.
* **CPE (Conditional Position Encoding):** Integrates dynamically generated positional information before graph construction to significantly enhance local feature representation.
* **MRConv4d (Max-Relative Graph Convolution):** Alleviates the over-smoothing problem in deep GCNs.
* **Edge-Optimized:** Demonstrated exceptional real-time inference latency (1.652 ms) and high mIoU (46.94%) on the **NVIDIA Jetson AGX Orin 64GB Developer Kit** via TensorRT.

The core network architecture can be found in `mvgc_backbone.py`.

---

## 🛠️ Installation & Dependencies

# Dynamic Multi-scale Vision Graph Convolution for Efficient Edge Perception in Autonomous Driving (MVGC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19660747.svg)](https://doi.org/10.5281/zenodo.19660747)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🔔 Important Notice:**
> This code repository is directly related to our manuscript currently under review at ***The Visual Computer***. If you find our work, code, or data helpful in your research, please consider citing our manuscript (see [Citation](#citation)).

---

## 📖 Abstract

Efficient visual perception is critical for safe autonomous driving on resource-constrained edge devices. Existing lightweight models struggle to balance global context and computational cost, as CNNs lack long-range dependencies while ViTs and graph neural networks suffer from high latency. This work presents **Multi-scale Vision Graph Convolution (MVGC)**, a hybrid architecture that eliminates KNN bottlenecks via dynamic sparse multi-scale graph attention and enhances local features with conditional position encoding. 

Experiments on MS COCO and BDD100K datasets show that MVGC improves object detection AP by 2.98% and instance segmentation AP by 11.51% with state-of-the-art accuracy-latency trade-off, making it suitable for real-time vehicular edge computing.

## 🚀 Highlights & Key Components

* **MSVGA (Multi-scale Vision Graph Attention):** Eliminates the computationally expensive KNN bottleneck found in traditional ViGs by utilizing a dynamic, sparse multi-scale connection strategy with linear complexity.
* **CPE :** Integrates dynamically generated positional information before graph construction to significantly enhance local feature representation.
* **Conv:** Alleviates the over-smoothing problem in deep GCNs.
* **Edge-Optimized:** Demonstrated exceptional real-time inference latency (1.652 ms) and high mIoU (46.94%) on the **NVIDIA Jetson AGX Orin 64GB Developer Kit** via TensorRT.

The core network architecture can be found in `mvgc_backbone.py`.

---

## 🛠️ Installation & Dependencies

Our framework is built upon PyTorch, [timm](https://github.com/rwightman/pytorch-image-models), and the [MMDetection](https://github.com/open-mmlab/mmdetection) ecosystem.

### Prerequisites
* Python >= 3.8
* PyTorch >= 1.12.0
* CUDA (Testing environment used NVIDIA GeForce RTX 5060 Ti)

### Setup
1. Clone this repository:
   ```bash
   git clone [https://github.com/lilithnvfsl-creator/MVGC.git](https://github.com/lilithnvfsl-creator/MVGC.git)
   cd MVGC

Data Preparation
Our experiments evaluate Object Detection and Instance Segmentation using the following datasets:

MS COCO 2017

BDD100K

ADE20K

Please download the datasets and organize them according to the standard MMDetection dataset structure in the data/ directory.


Training
To train the MVGC model from scratch or fine-tune it, use the main.py script. Make sure to specify the path to your configuration file.
# Single GPU training
python main.py configs/mvgc/mvgc_m_coco.py --work-dir ./work_dirs/mvgc_m

# Multi-GPU Distributed Training
python -m torch.distributed.launch --nproc_per_node=4 main.py configs/mvgc/mvgc_m_coco.py --launcher pytorch

Evaluation (Testing)
To evaluate a trained MVGC checkpoint on the test set, use the test.py script:
python test.py configs/mvgc/mvgc_m_coco.py ./work_dirs/mvgc_m/latest.pth --eval bbox segm
