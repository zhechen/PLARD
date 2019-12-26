# PLARD 

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/zhechen/PLARD/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.1109/JAS.2019.1911459.svg)](https://doi.org/10.1109/JAS.2019.1911459)



## Progressive LiDAR Adaptation for Road Detection Implemented in PyTorch

This repository reproduces results of PLARD in PyTorch. The code is heavily based on [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).


<p align="center">
<img src="imgs/main.png" width="80%"/>
</p>

### Abstract

Despite rapid developments in visual image-based road detection, robustly identifying road areas in visual images remains challenging due to issues like illumination changes and blurry images. To this end, LiDAR sensor data can be incorporated to improve the visual image-based road detection, because LiDAR data is less susceptible to visual noises. However, the main difficulty in introducing LiDAR information into visual image-based road detection is that LiDAR data and its extracted features do not share the same space with the visual data and visual features. Such gaps in spaces may limit the benefits of LiDAR information for road detection. To overcome this issue, we introduce a novel Progressive LiDAR Adaptation-aided Road Detection (PLARD) approach to adapt LiDAR information into visual image-based road detection and improve detection performance. In PLARD, progressive LiDAR adaptation consists of two subsequent modules: 1) data space adaptation, which transforms the LiDAR data to the visual data space to align with the perspective view by applying altitude difference-based transformation; and 2) feature space adaptation, which adapts LiDAR features to visual features through a cascaded fusion structure. Comprehensive empirical studies on the well-known KITTI road detection benchmark demonstrate that PLARD takes advantage of both the visual and LiDAR information, achieving much more robust road detection even in challenging urban scenes. In particular, PLARD outperforms other state-of-the-art road detection models and is currently top of the publicly accessible benchmark leader-board.


## Usage

### Setup



### Usage

**Setup config file**

```yaml
# Model Configuration
model:
    arch: <name> [options: 'fcn[8,16,32]s, unet, segnet, pspnet, icnet, icnetBN, linknet, frrn[A,B]'
    <model_keyarg_1>:<value>

# Data Configuration
data:
    dataset: <name> [options: 'pascal, camvid, ade20k, mit_sceneparsing_benchmark, cityscapes, nyuv2, sunrgbd, vistas'] 
    train_split: <split_to_train_on>
    val_split: <spit_to_validate_on>
    img_rows: 512
    img_cols: 1024
    path: <path/to/data>
    <dataset_keyarg1>:<value>

# Training Configuration
training:
    n_workers: 64
    train_iters: 35000
    batch_size: 16
    val_interval: 500
    print_interval: 25
    loss:
        name: <loss_type> [options: 'cross_entropy, bootstrapped_cross_entropy, multi_scale_crossentropy']
        <loss_keyarg1>:<value>

    # Optmizer Configuration
    optimizer:
        name: <optimizer_name> [options: 'sgd, adam, adamax, asgd, adadelta, adagrad, rmsprop']
        lr: 1.0e-3
        <optimizer_keyarg1>:<value>

        # Warmup LR Configuration
        warmup_iters: <iters for lr warmup>
        mode: <'constant' or 'linear' for warmup'>
        gamma: <gamma for warm up>
       
    # Augmentations Configuration
    augmentations:
        gamma: x                                     #[gamma varied in 1 to 1+x]
        hue: x                                       #[hue varied in -x to x]
        brightness: x                                #[brightness varied in 1-x to 1+x]
        saturation: x                                #[saturation varied in 1-x to 1+x]
        contrast: x                                  #[contrast varied in 1-x to 1+x]
        rcrop: [h, w]                                #[crop of size (h,w)]
        translate: [dh, dw]                          #[reflective translation by (dh, dw)]
        rotate: d                                    #[rotate -d to d degrees]
        scale: [h,w]                                 #[scale to size (h,w)]
        ccrop: [h,w]                                 #[center crop of (h,w)]
        hflip: p                                     #[flip horizontally with chance p]
        vflip: p                                     #[flip vertically with chance p]

    # LR Schedule Configuration
    lr_schedule:
        name: <schedule_type> [options: 'constant_lr, poly_lr, multi_step, cosine_annealing, exp_lr']
        <scheduler_keyarg1>:<value>

    # Resume from checkpoint  
    resume: <path_to_checkpoint>
```

**To train the model :**

```
python train.py [-h] [--config [CONFIG]] 

--config                Configuration file to use
```

**To validate the model :**

```
usage: validate.py [-h] [--config [CONFIG]] [--model_path [MODEL_PATH]]
                       [--eval_flip] [--measure_time]

  --config              Config file to be used
  --model_path          Path to the saved model
  --eval_flip           Enable evaluation with flipped image | True by default
  --measure_time        Enable evaluation with time (fps) measurement | True
                        by default
```


**If you find this code useful in your research, please consider citing:**

```
@article{mshahsemseg,
    Author = {Meet P Shah},
    Title = {Semantic Segmentation Architectures Implemented in PyTorch.},
    Journal = {https://github.com/meetshah1995/pytorch-semseg},
    Year = {2017}
}
```

