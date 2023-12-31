# Celestial Object Recognition using Feature Pyramid Network

![image](https://github.com/prathmeshlonkar10/Celestial-object-recognition-using-Feature-Pyramid-Network/assets/66990159/f0d3c8b1-2b6a-4987-81bf-44e7eb11dc40)

## Highlight
We are given a data synthesizer that generates images and labels. The goal is to train a model with at most 4.5 million trainable parameters which determines whether each image has a star and, if so, finds a rotated bounding box that bounds the star (as shown in the example.png file).

More precisely, the labels contain the following five numbers, which the model should predict:
* the x and y coordinates of the center
* yaw
* width and height.

If there is no star, the label consists of 5 `np.nan`s. The height of the star is always noticeably larger than its width, and the yaw points in one of the height directions. The yaw is always in the interval `[0, 2 * pi)`, oriented counter-clockwise and with zero corresponding to the upward direction. train.py contains a basic CNN architecture (and training code) that performs fairly and you can extend this model/training or start over on your own.
