# Happy or Sad Image Classifier
*<b>Summary</b>: This project holds within a model built of a convolutional neural network, capable of classifying a provided input image as that of a sad person or happy person.*

## Overview

This repository contains a classification model that takes an image input and classifies it as an image of a sad person or a happy person. The model is built using TensorFlow version 2.11.0 and Python version 3.7.

## Model Architecture

The neural network architecture used in this classification model is as follows:

1. **Input Layer**: The neural network accepts images with a size of 256x256 pixels and 3 colour channels (RGB) as input. Each pixel represents a tiny dot with different colour information.

2. **Convolutional Layers**: The first step is to apply a set of filters (like looking through different lenses) to the input image. Each filter is a small window that slides across the image, scanning for specific features, patterns, or edges. In this network, we have three sets of convolutional layers:
   - The first convolutional layer consists of 16 filters (lenses) with a size of 3x3 pixels. These filters are sensitive to various visual patterns, and when they detect something interesting in the image, they highlight it by activating (using the `relu` activation function).
   - Next, the first `MaxPooling` layer reduces the spatial size of the image. It takes the maximum value in each small region of the image, essentially downsizing it and capturing the most significant features.
   - The second convolutional layer has 32 filters with a 3x3 size. It looks for more complex features and patterns in the reduced image from the first `MaxPooling`.
   - The second `MaxPooling` layer further reduces the image size while preserving essential features.
   - The third convolutional layer, similar to the first one, has 16 filters and continues to detect more intricate patterns after the second `MaxPooling`.

3. **Flatten Layer**: After the convolutional layers and `MaxPooling`, we "flatten" the output. This means we transform the 2D feature maps into a 1D vector, preparing it for the next step.

4. **Dense Layers (Fully Connected)**: Now, the network enters a more traditional neural network structure. The Dense layers consist of nodes (neurons) that are fully connected. This allows the network to learn complex relationships between the features identified in the previous layers.
   - The first Dense layer has 256 neurons and uses the `relu` activation function. This layer is responsible for finding higher-level patterns in the flattened features.
   - The final Dense layer has only one neuron, and it uses the `sigmoid` activation function. This is the output layer that predicts a single value, either close to 0 (representing one class) or close to 1 (representing the other class). In this case, the network is likely performing a binary classification task, deciding between two classes.

## Model Prediction

The neural network outputs a probability between 0 and 1, representing the likelihood of the input image belonging to the class of a sad person. If the probability of a sad person is higher than a threshold (threshold = 0.5), the image is classified as a sad person's image. Otherwise, it is classified as a happy person's image.

## Usage

To use this model, you can follow these steps:

1. Ensure you have Python 3.7 and TensorFlow 2.11.0 installed.

2. Clone this repository to your local machine.

3. Prepare your image data for testing. The input images should have a size of 256x256 pixels and 3 colour channels (RGB).

4. Run the model on your test data to get predictions.

```python
# Example code for loading an image and making predictions

 # Import the dependencies:
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

model = load_model('model/happy_sad_classifier.h5') # load the model

your_img = cv2.imread('image_directory.jpg') # reads your image
your_resized_img = tf.image.resize(your_img, (256,256)) # resizes the image for compatibility

yhat = model.predict(np.expand_dims(your_resized_img/255, 0)) # runs the model to predict the probability and stores it in 'yhat'

if yhat > 0.5: # 0.5 is the threshold
    print(f'Person(s) in the picture is Sad :(') # if the probability is equal-lower than 0.5 it is a sad person
else:
    print(f'Person(s) in the picture is Happy :)') # if the probability is higher than 0.5 it is a happy person

```

Remember to replace `image_directory.jpg` and `model/happy_sad_classifier.h5` with the appropriate file paths, though if you clone the repository the model path should be fine as it is.

*Speacial thanks to* @nicknochnack *for providing his project on https://github.com/nicknochnack/ImageClassification.git. This project is heavily inspired by it.*
