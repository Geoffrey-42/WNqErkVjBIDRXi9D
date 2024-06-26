# Video Event Detection and Action Identification
This work is part of the development of a new mobile application for the blind or anyone who wishes to scan and read document in bulk.
All the user has to do is to flip pages and the phone camera will automatically detect the page flipping event, scan the new pages and perform an OCR.

## Objective
The goal is to detect flipping events in videos or sequences of images of pages seen from a phone camera.

## Dataset
A dataset of images labeled as "flip" or "notflip" is used, containing pictures of pages being flipped or not flipped.

## Methodology
The selected approach involves training a model to classify images of pages based on whether they are being flipped.

Using Tensorflow, transfer learning techniques were employed to fine-tune existing Convolutional Neural Network (CNN) architectures such as VGG16, MobileNetv2, and ResNet50.

Additionally, a custom lightweight CNN architecture was specifically designed for this task. 
The much smaller size of the model results in faster predictions and is more adapted for a use on a mobile phone with limited storage space and computational capacities.

The custom model was prototyped using Gradio, resulting in a permanent interface that can be accessed through a web browser. Through this web interface, users can conveniently utilize the models and obtain predictions.

## Results
High accuracy of the models in predicting page flipping actions was observed on the test set, showcasing their effectiveness.

## Features

- Automatic detection of page flipping actions using single images
- Detection of flipping actions in a sequence of images within a video
- Utilizes Tensorflow framework
- Fine-tuned pre-trained CNN architectures (VGG16, MobileNetv2, and ResNet50)
- Custom lightweight CNN architecture designed specifically for page flipping detection
- Permanent interface accessible through a web browser using Gradio
- High accuracy in predicting page flipping actions

## Getting Started

- Open the VideoEventDetection folder
- Click on the "Gradio App Link" file
- Copy paste the link on your browser and try the prototype!

## Conclusion
Through the application of either transfer learning or a custom lightweight CNN architecture, the mobile app can achieve fully automatic and fast document scanning, rendering it a valuable tool for diverse user groups.


