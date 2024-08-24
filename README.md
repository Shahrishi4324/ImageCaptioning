# Image Captioning with Attention Mechanism

## Project Overview

This project demonstrates how to generate captions for images using a deep learning model that integrates both Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) with an attention mechanism for sequence prediction. The model is trained on image-caption pairs and is capable of generating descriptive captions for new images.

## Key Features

- **Image Feature Extraction**: Uses a pre-trained CNN (ResNet) as an encoder to extract image features.
- **Attention Mechanism**: Implements an attention mechanism that allows the model to focus on different parts of the image when generating each word of the caption.
- **Caption Generation**: Uses an RNN decoder to generate captions word by word, leveraging the attention mechanism to enhance descriptive accuracy.

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
  
## Installation

To run this project, you'll need to have Python installed along with the following libraries:

```bash
pip install torch torchvision matplotlib
```

Clone the repository: 
```bash
git clone https://github.com/Shahrishi4324/ImageCaptioning.git
cd ImageCaptioning
```

## Datasets: 
This project uses the COCO dataset for image captioning. You need to download the images and the captions file:

Images: Download the COCO images from the COCO dataset website. You need the train2014 and/or val2014 images. Extract these into a directory and set the image_folder variable in the code to the path of this directory.

Captions: Download the captions JSON file from the COCO dataset website. You need the annotations_train2014.json and/or annotations_val2014.json file. Set the caption_file variable in the code to the path of this file.

## Usage: 
1. Prepare the dataset: Ensure you have the COCO dataset images and captions downloaded and placed in the coco/ and captions/ directories respectively. Update the image_folder and caption_file variables in the code with the correct paths.
2. Run the Project: Execute the main script to start training the model:
```bash
python ImageCaptioning.py
```
