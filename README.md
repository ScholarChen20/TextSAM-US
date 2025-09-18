# TextSAM-US: Text Prompt Learning for SAM to Accurately Segment Breast Tumor in Ultrasound Images

[//]: # (**[Health-X Lab]&#40;http://www.healthx-lab.ca/&#41;** | **[IMPACT Lab]&#40;https://users.encs.concordia.ca/~impact/&#41;** )

[//]: # ([Pascal Spiegler]&#40;https://scholar.google.com/citations?user=FoihFT0AAAAJ&hl=en&#41;, [Taha Koleilat]&#40;https://tahakoleilat.github.io/&#41;, [Arash Harirpoush]&#40;https://scholar.google.com/citations?user=-jhPnlgAAAAJ&hl=en&#41;, [Corey S. Miller]&#40;https://www.mcgill.ca/gastroenterology/corey-miller&#41;, [Hassan Rivaz]&#40;https://users.encs.concordia.ca/~hrivaz/&#41;, [Marta Kersten-Oertel]&#40;https://www.martakersten.ca/&#41;, [Yiming Xiao]&#40;https://yimingxiao.weebly.com/curriculum-vitae.html&#41;)

[//]: # ([![paper]&#40;https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg&#41;]&#40;https://www.arxiv.org/abs/2507.18082&#41;)

[//]: # ([![Overview]&#40;https://img.shields.io/badge/Overview-Read-blue.svg&#41;]&#40;#overview&#41;)

[//]: # ([![Datasets]&#40;https://img.shields.io/badge/Datasets-Access-yellow.svg&#41;]&#40;https://drive.google.com/drive/folders/10GPl3r-ppDyWwWzneoSFH52yxUGX4xkw&#41;)

[//]: # ([![Checkpoint]&#40;https://img.shields.io/badge/Models-Reproduce-orange.svg&#41;]&#40;https://drive.google.com/file/d/152U8ZilljXfGSqwN3m77JQpKLXR7mZR0/view&#41;)

[//]: # ([![BibTeX]&#40;https://img.shields.io/badge/BibTeX-Cite-blueviolet.svg&#41;]&#40;#citation&#41;)

## Overview

[//]: # (> **<p align="justify"> Abstract:** *Pancreatic cancer carries a poor prognosis and relies on endoscopic ultrasound &#40;EUS&#41; for targeted biopsy and radiotherapy. However, the speckle noise, low contrast, and unintuitive appearance of EUS make segmentation of pancreatic tumors with fully supervised deep learning &#40;DL&#41; models both error-prone and dependent on large, expert-curated annotation datasets. To address these challenges, we present **TextSAM-EUS**, a novel, lightweight, text-driven adaptation of the Segment Anything Model &#40;SAM&#41; that requires no manual geometric prompts at inference. Our approach leverages text prompt learning &#40;context optimization&#41; through the BiomedCLIP text encoder in conjunction with a LoRA-based adaptation of SAM’s architecture to enable automatic pancreatic tumor segmentation in EUS, tuning only 0.86% of the total parameters. On the public Endoscopic Ultrasound Database of the Pancreas, **TextSAM-EUS** with automatic prompts attains 82.69% Dice and 85.28% normalized surface distance &#40;NSD&#41;, and with manual geometric prompts reaches 83.10% Dice and 85.70% NSD, outperforming both existing state-of-the-art &#40;SOTA&#41; supervised DL models and foundation models &#40;e.g., SAM and its variants&#41;. As the first attempt to incorporate prompt learning in SAM-based medical image segmentation, **TextSAM-EUS** offers a practical option for efficient and robust automatic EUS segmentation. Our code will be publicly available upon acceptance.* </p>)

## Method


## Segmentation Results


## Installation 
This codebase is tested on Ubuntu 20.04.2 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n textsam_eus python=3.10 -y

# Activate the environment
conda activate textsam_eus

# Install torch (requires version >= 2.1.2) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

```
* Clone TextSAM-EUS code repository and install requirements
```bash
# Clone MaPLe code base
git clone https://github.com/HealthX-Lab/TextSAM-EUS

cd TextSAM-EUS/
# Install requirements

pip install -e .
```

## Data preparation

* Download the dataset [here]().

* Place dataset under `data` like the following:
```
data/
|–– EUS/
|   |–– train/
|   |   |–– images/
|   |   |–– masks/
|   |–– val/
|   |   |–– images/
|   |   |–– masks/
|   |–– test/
|   |   |–– images/
|   |   |–– masks/
```

## Training and Evaluation
* Run the training and evaluation script

```bash
bash scripts/pipeline.sh EUS outputs
```

* You can change some design settings in the [config](https://github.com/HealthX-Lab/PanTumorUSSeg/blob/main/configs/EUS.yaml).

## Citation
If you use our work, please consider citing:

[//]: # (```bibtex)

[//]: # (@article{spiegler2025textsam,)

[//]: # (  title={TextSAM-EUS: Text Prompt Learning for SAM to Accurately Segment Pancreatic Tumor in Endoscopic Ultrasound},)

[//]: # (  author={Spiegler, Pascal and Koleilat, Taha and Harirpoush, Arash and Miller, Corey S and Rivaz, Hassan and Kersten-Oertel, Marta and Xiao, Yiming},)

[//]: # (  journal={arXiv preprint arXiv:2507.18082},)

[//]: # (  year={2025})

[//]: # (})
```

## Acknowledgements

Our code builds upon the [open_clip](https://github.com/mlfoundations/open_clip), [segment-anything](https://github.com/facebookresearch/segment-anything), and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) repositories. We are grateful to the authors for making their code publicly available. If you use our model or code, we kindly request that you also consider citing these foundational works.
