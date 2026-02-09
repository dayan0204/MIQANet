# MIQANet: A Novel Dual-Branch Deep Learning Framework for MRI Image Quality Assessment

[Link to the paper](https://ieeexplore.ieee.org/document/11360772)

## Abstract
Image quality assessment (IQA) algorithms have significantly advanced over the past two decades, primarily focusing on natural images. However, applying these methods directly to medical imaging often yields suboptimal performance due to inherent differences such as the structural complexity of medical images and the limited availability of annotated databases. In this study, we conduct a comprehensive evaluation of state-of-the-art IQA methods, including 29 traditional full-reference (FR), 4 traditional no-reference (NR), and 9 deep learning-based approaches, to assess their effectiveness in the context of medical imaging. Our evaluation is performed on a recently developed MRI image quality assessment benchmark, revealing critical performance gaps in existing methods. Building on these findings, we propose a novel dual-branch deep learning framework specifically designed for medical IQA (MIQANet). The proposed approach effectively combines global contextual information with local structural details, enhancing the model’s ability to capture subtle degradations and structural inconsistencies in MRI scans. Experiential results demonstrate the superiority of our approach over existing methods, providing valuable theoretical and practical insights for enhancing quality assessment of medical images.

## Network Architecture
![Network Architecture](Network/Overall_architecture_new.jpg)

## Dataset

MIQANet is developed and evaluated using the **RAD-IQMRI** radiologist-rated MRI image quality assessment benchmark dataset.

RAD-IQMRI GitHub repository:  
https://github.com/dayan0204/RAD-IQMRI

If you use MIQANet together with this dataset, please also cite the RAD-IQMRI dataset paper.

---

## **USAGE**
```sh
python train.py
```

## **Requirements**
```sh
pip install -r requirements.txt
```

## **Citation**
```sh
@ARTICLE{11360772,
  author={Ma, Yueran and Wang, Huasheng and Wu, Yingying and Tanguy, Jean-Yves and White, Richard and Wardle, Phillip and Krupinski, Elizabeth and Corcoran, Padraig and Liu, Hantao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={MIQANet: A Novel Dual-Branch Deep Learning Framework for MRI Image Quality Assessment}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Magnetic resonance imaging;Image quality;Measurement;Quality assessment;Noise;Degradation;Visualization;Medical diagnostic imaging;Deep learning;Computational modeling;Image quality assessment;medical image;deep learning;artifacts;MRI},
  doi={10.1109/TCSVT.2026.3656671}}
```

## **Acknowledgment**
Our code incorporates components and design ideas inspired by [MANIQA](https://ieeexplore.ieee.org/document/10539107)
