# Celestial Object Recognition using Feature Pyramid Network

![image](https://github.com/prathmeshlonkar10/Celestial-object-recognition-using-Feature-Pyramid-Network/assets/66990159/f0d3c8b1-2b6a-4987-81bf-44e7eb11dc40)

## Highlight
We are given a data synthesizer that generates images and labels. The goal is to train a model with at most 4.5 million trainable parameters which determines whether each image has a star and, if so, finds a rotated bounding box that bounds the star (as shown below).

![image](https://github.com/prathmeshlonkar10/Celestial-object-recognition-using-Feature-Pyramid-Network/assets/66990159/d3e765a6-9578-4ff1-959c-1a0cd7b1f809)

More precisely, the labels contain the following five numbers, which the model should predict:
* the x and y coordinates of the center
* yaw
* width and height.

If there is no star, the label consists of 5 `np.nan`s. The height of the star is always noticeably larger than its width, and the yaw points in one of the height directions. The yaw is always in the interval `[0, 2 * pi)`, oriented counter-clockwise and with zero corresponding to the upward direction. train.py contains a basic CNN architecture (and training code) that performs fairly and you can extend this model/training or start over on your own.

## Technologies used
![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![](https://camo.githubusercontent.com/58bfe5f46be0cf6c7d0b34f17a83ad69250fc9180ef95018eacfd283cdc61c10/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d6174706c6f746c69622d3243324437323f7374796c653d666f722d7468652d6261646765266c6f676f3d6d6174706c6f746c6962266c6f676f436f6c6f723d7768697465)
