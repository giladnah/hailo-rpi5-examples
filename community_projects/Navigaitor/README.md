## NavigAItor: AI That Knows The Way
<!-- [Guilherme Potje](https://guipotje.github.io/) · [Felipe Cadar](https://eucadar.com/) · [Andre Araujo](https://andrefaraujo.github.io/) · [Renato Martins](https://renatojmsdh.github.io/) · [Erickson R. Nascimento](https://homepages.dcc.ufmg.br/~erickson/) -->

<!-- [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_matching.ipynb) -->

<!-- ### [[ArXiv]](https://arxiv.org/abs/2404.19174) | [[Project Page]](https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/) |  [[CVPR'24 Paper]](https://cvpr.thecvf.com/)

<div align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
  <div style="display: flex; justify-content: space-around; width: 100%;">
    <img src='./figs/xfeat.gif' width="400"/>
    <img src='./figs/sift.gif' width="400"/>
  </div>
  
  Real-time XFeat demonstration (left) compared to SIFT (right) on a textureless scene. SIFT cannot handle fast camera movements, while XFeat provides robust matches under adverse conditions, while being faster than SIFT on CPU.
  
</div>

**TL;DR**: Really fast learned keypoint detector and descriptor. Supports sparse and semi-dense matching.

Just wanna quickly try on your images? Check this out: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_torch_hub.ipynb)

## Table of Contents
- [Introduction](#introduction) <img align="right" src='./figs/xfeat_quali.jpg' width=360 />
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Real-time demo app](#real-time-demo)
- [Contribute](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements) -->

## Introduction
Introducing NavigAItor – a simple yet powerful robot that autonomously navigates along a pre-recorded path, moving anywhere you need it - forward, backward, left and right, with just the press of a button, No GPS, no Jiro, just pure intelligence.
By capturing images and creating a video-based database, NavigAItor builds a path story, much like a person walking a trail and marking key points to remember the way back. 
The neural network combines interest point detection and descriptor extraction into a single model
<!-- In the first step, and just once, you’ll define manually the path you want the robot to follow.
The robot will capture this route, an image per X, creating the foundation for seamless, automatic navigation in the future.
Once the path is set, it's that simple – just place the robot at any point along the route. With a single button press, the robot will effortlessly move forward, backward, stop, and return along the path as needed – smart, precise, and completely hands-free. -->

**Motivation.** 

**Capabilities.**
- Real-time sparse inference on CPU for VGA images (tested on laptop with an i5 CPU and vanilla pytorch);
- Simple architecture components which facilitates deployment on embedded devices (jetson, raspberry pi, hailo 8, etc..);
- Supports both sparse and semi-dense matching of local features;
- Compact descriptors (64D);
- Performance comparable to known deep local features such as SuperPoint while being significantly faster and more lightweight. Also, XFeat exhibits much better robustness to viewpoint and illumination changes than classic local features as ORB and SIFT;
- Supports batched inference if you want ridiculously fast feature extraction. On VGA sparse setting, we achieved about 1,400 FPS using an RTX 4090.
- For single batch inference on GPU (VGA), one can easily achieve over 150 FPS while leaving lots of room on the GPU for other concurrent tasks.

##

**Paper Abstract.** We introduce a lightweight and accurate architecture for resource-efficient visual correspondence. Our method, dubbed XFeat (Accelerated Features), revisits fundamental design choices in convolutional neural networks for detecting, extracting, and matching local features. Our new model satisfies a critical need for fast and robust algorithms suitable to resource-limited devices. In particular, accurate image matching requires sufficiently large image resolutions -- for this reason, we keep the resolution as large as possible while limiting the number of channels in the network. Besides, our model is designed to offer the choice of matching at the sparse or semi-dense levels, each of which may be more suitable for different downstream applications, such as visual navigation and augmented reality. Our model is the first to offer semi-dense matching efficiently, leveraging a novel match refinement module that relies on coarse local descriptors. XFeat is versatile and hardware-independent, surpassing current deep learning-based local features in speed (up to 5x faster) with comparable or better accuracy, proven in pose estimation and visual localization. We showcase it running in real-time on an inexpensive laptop CPU without specialized hardware optimizations.

**Overview of XFeat's achitecture.**
XFeat extracts a keypoint heatmap $\mathbf{K}$, a compact 64-D dense descriptor map $\mathbf{F}$, and a reliability heatmap $\mathbf{R}$. It achieves unparalleled speed via early downsampling and shallow convolutions, followed by deeper convolutions in later encoders for robustness. Contrary to typical methods, it separates keypoint detection into a distinct branch, using $1 \times 1$ convolutions on an $8 \times 8$ tensor-block-transformed image for fast processing, being one of the few current learned methods that decouples detection & description and can be processed independently.

<img align="center" src="./figs/xfeat_arq.png" width=1000 />

## Installation
<!-- XFeat has minimal dependencies, only relying on torch. Also, XFeat does not need a GPU for real-time sparse inference (vanilla pytorch w/o any special optimization), unless you run it on high-res images. If you want to run the real-time matching demo, you will also need OpenCV.
We recommend using conda, but you can use any virtualenv of your choice.
If you use conda, just create a new env with: -->
```bash
git clone https://github.com/eilonpo/navigaitor/
cd community_projects/Navigaitor/

# setup web server for interaction with Raspbot:
sudo apt install pipx 
pipx install poetry 
export PATH="$HOME/.local/bin:$PATH" #add bin dir to path
poetry install #will create the venv and install all deps, then use poetry shell

#create & activate hailo env
. ../../setup_env.sh
pip install torch opencv-python onnxruntime tdqm

# get the Raspbot ip address from Raspbot display
# open a browser and connect to the Raspbot server
# <ip_addr>:8000
```

## Usage
use the web GUI to control the Raspbot:
1. start recording mode and navigate to the desired path
2. stop recording
3. move the robot back to starting point using the navigation buttons
4. start retreat mode and watch the robot repeats the track!

