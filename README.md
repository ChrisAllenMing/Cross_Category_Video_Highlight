# Cross-category Video Highlight Detection via Set-based Learning

## Introduction

This demo project is an implementation of ``Cross-category Video Highlight Detection via Set-based Learning'' in PyTorch. 
We provide the codes for *SL-module* and *DL-VHD* on YouTube Highlights dataset. 

### Prerequisites

We develop this project with `Python3.6` and following Python packages:
```
Pytorch                   1.5.0
cv2                       3.4.2
einops                    0.3.0
```
**P.S.** In our project, these packages can be successfully installed and work together under `CUDA/9.0` and `cuDNN/7.0.5`.

### Dataset and Pre-trained Model

**Dataset.** This demo project only includes the implementation on YouTube Highlights dataset, you can download the dataset in [this pervious work](https://github.com/aliensunmin/DomainSpecificHighlight) and put it under the path you like, e.g. `~/data/YouTube_Highlights/`. 

**Dataset pre-processing.** For training highlight detection models, you can convert the original videos in YouTube Highlights to video segments by following command:
```
python ./dataloaders/youtube_highlights_set.py
```

**Pre-trained model.** You can download the pre-trained C3D model in [this url](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing) and put it under the path you like, e.g. `~/pretrained_models/`.

### Category-specific Video Highlight Detection

**SL-module.** To train (also finally evaluate) the SL-module, simply run: 
```
python train_SL_module.py --gpu_id $device_id$ --src_category $cls$ \
                          --tgt_category $cls$ --use_transformer
```

### Cross-category Video Highlight Detection

**Source-only.** To train (also finally evaluate) the Source-only model with SL-module, simply run:
```
python train_SL_module.py --gpu_id $device_id$ --src_category $src_cls$ \
                          --tgt_category $tgt_cls$ --use_transformer
```

**Target-oracle.** To train (also finally evaluate) the Target-oracle model with SL-module, simply run:
```
python train_SL_module.py --gpu_id $device_id$ --src_category $tgt_cls$ \
                          --tgt_category $tgt_cls$ --use_transformer
```

**DL-VHD.** To train (also finally evaluate) the DL-VHD model, simply run:
```
python train_dual_learner.py --gpu_id $device_id$ --src_category $src_cls$ \
                             --tgt_category $tgt_cls$ --use_transformer
```