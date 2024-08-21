# Image Captioning with Attention Mechanism

## Project Overview

This project demonstrates how to generate captions for images using a deep learning model that integrates both Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) with an attention mechanism for sequence prediction. The model is trained on image-caption pairs and is capable of generating descriptive captions for new images.

## Key Features

- **Image Feature Extraction**: Uses a pre-trained CNN (ResNet) as an encoder to extract image features.
- **Attention Mechanism**: Implements an attention mechanism that allows the model to focus on different parts of the image when generating each word of the caption.
- **Caption Generation**: Uses an RNN decoder to generate captions word by word, leveraging the attention mechanism to enhance descriptive accuracy.

## Installation

To run this project, you'll need to have Python installed along with the following libraries:

```bash
pip install torch torchvision matplotlib
```

Run the project with: 
```bash
python ImageCaptioning.py
```
