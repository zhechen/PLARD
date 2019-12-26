# PLARD 

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/zhechen/PLARD/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.1109/JAS.2019.1911459.svg)](https://doi.org/10.1109/JAS.2019.1911459)



## Progressive LiDAR Adaptation for Road Detection Implemented in PyTorch

This repository reproduces the results of PLARD [PDF](https://arxiv.org/abs/1904.01206) in PyTorch. The code is heavily based on [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).


<p align="center">
<img src="imgs/main.png" width="95%"/>
</p>

### Abstract

Despite rapid developments in visual image-based road detection, robustly identifying road areas in visual images remains challenging due to issues like illumination changes and blurry images. To this end, LiDAR sensor data can be incorporated to improve the visual image-based road detection, because LiDAR data is less susceptible to visual noises. However, the main difficulty in introducing LiDAR information into visual image-based road detection is that LiDAR data and its extracted features do not share the same space with the visual data and visual features. Such gaps in spaces may limit the benefits of LiDAR information for road detection. To overcome this issue, we introduce a novel Progressive LiDAR Adaptation-aided Road Detection (PLARD) approach to adapt LiDAR information into visual image-based road detection and improve detection performance. In PLARD, progressive LiDAR adaptation consists of two subsequent modules: 1) data space adaptation, which transforms the LiDAR data to the visual data space to align with the perspective view by applying altitude difference-based transformation; and 2) feature space adaptation, which adapts LiDAR features to visual features through a cascaded fusion structure. Comprehensive empirical studies on the well-known KITTI road detection benchmark demonstrate that PLARD takes advantage of both the visual and LiDAR information, achieving much more robust road detection even in challenging urban scenes. In particular, PLARD outperforms other state-of-the-art road detection models and is currently top of the publicly accessible benchmark leader-board.


### Installation ###
Please follow instructions on [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

### Setup

**Setup Dataset**

Please setup dataset according to the following folder structure:
```
PLARD
 |---- ptsemseg
 |---- imgs
 |---- outputs
 |---- dataset
 |    |---- training
 |    |    |---- image_2
 |    |    |---- ADI
 |    |---- testing
 |    |    |---- image_2
 |    |    |---- ADI 
```
The "image\_2" folders contain the visual images which can be downloaded from the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php).

The "ADI" folders contain the altitude difference images which can be downloaded [here](https://www.dropbox.com/s/wks807hv84wcduv/ADI-training.zip?dl=0) for training set and [here](https://www.dropbox.com/s/sslqw2flp7ptwnj/ADI-testing.zip?dl=0) for testing set. 

### Usage
**Test**

Run the test set on KITTI Road dataset using the following command:
```
python test.py --model_path /path/to/plard_kitti_road.pth
```
The results in perspective view will be written under "./outputs/results". Follow the guidelines of KITTI to perform evaluation. 

A trained model of PLARD can be downloaded [here](https://www.dropbox.com/s/lrtwhuo0rxv1sgy/plard_kitti_road.pth?dl=0). 

**Train**

Training script is similar to the [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

### Reference
**If you find this code useful in your research, please consider citing:**

```
@article{chen2019progressive,
  title={Progressive LiDAR adaptation for road detection},
  author={Chen, Zhe and Zhang, Jing and Tao, Dacheng},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={6},
  number={3},
  pages={693--702},
  year={2019},
  publisher={IEEE}
}
```

