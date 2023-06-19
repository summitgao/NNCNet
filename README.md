**PyTorch implementation of [Nearest Neighbor-Based Contrastive Learning for Hyperspectral and LiDAR Data Classification](https://ieeexplore.ieee.org/abstract/document/10015054) in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-16, 2023, Art no. 5501816, doi: 10.1109/TGRS.2023.3236154.**

If you have any questions, please contact us. Email: <a href="wm@stu.ouc.edu.cn">wm@stu.ouc.edu.cn</a>, <a href="gaofeng@ouc.edu.cn">gaofeng@ouc.edu.cn</a>

## Introduction

The joint hyperspectral image (HSI) and light detection and ranging (LiDAR) data classification aims to interpret ground objects at more detailed and precise level. Although deep learning methods have shown remarkable success in the multisource data classification task, self-supervised learning has rarely been explored. It is commonly nontrivial to build a robust self-supervised learning model for multisource data classification, due to the fact that the semantic similarities of neighborhood regions are not exploited in the existing contrastive learning framework. Furthermore, the heterogeneous gap induced by the inconsistent distribution of multisource data impedes the classification performance. To overcome these disadvantages, we propose a nearest neighbor-based contrastive learning network (NNCNet), which takes full advantage of large amounts of unlabeled data to learn discriminative feature representations. Specifically, we propose a nearest neighbor-based data augmentation scheme to use enhanced semantic relationships among nearby regions. The intermodal semantic alignments can be captured more accurately. In addition, we design a bilinear attention module to exploit the second-order and even high-order feature interactions between the HSI and LiDAR data. Extensive experiments on four public datasets demonstrate the superiority of our NNCNet over state-of-the-art methods. 

## Dataset

1. MUUFL dataset: https://github.com/GatorSense/MUUFLGulfport
2. Houston 2013 dataset: https://hyperspectral.ee.uh.edu/?page_id=459
3. Houston 2018 dataset: https://hyperspectral.ee.uh.edu/?page_id=1075


