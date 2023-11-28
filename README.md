# MonReader

MonReader has been designed as a new mobile document digitization experience for the blind, researchers, and anyone seeking fully automatic, fast, and high-quality document scanning in bulk. All scanning tasks are handled by a mobile app.

The main objective of MonReader is to detect if a page is being flipped using a single image and determine whether a sequence of images in a video contains the action of flipping. To accomplish this, the Tensorflow framework was utilized, and transfer learning techniques were employed to fine-tune existing Convolutional Neural Network (CNN) architectures such as VGG16, MobileNetv2, and ResNet50.

Additionally, a custom lightweight CNN architecture was specifically designed for this task. The custom model was prototyped using Gradio, resulting in a permanent interface that can be accessed through a web browser. Through this web interface, users can conveniently utilize the models and obtain predictions.

Extensive testing has demonstrated the high accuracy of the models in predicting page flipping actions on the test set, showcasing their effectiveness. The dual use of transfer learning and the custom lightweight CNN architecture allows MonReader to achieve fully automatic, fast, and high-quality document scanning, rendering it a valuable tool for diverse user groups.

## Features

- Automatic detection of page flipping actions using single images
- Detection of flipping actions in a sequence of images within a video
- Utilizes Tensorflow framework
- Fine-tuned pre-trained CNN architectures (VGG16, MobileNetv2, and ResNet50)
- Custom lightweight CNN architecture designed specifically for page flipping detection
- Permanent interface accessible through a web browser using Gradio
- High accuracy in predicting page flipping actions

## Getting Started

- Open the MonReader folder
- Click on the Gradioapp link folder
- Copy paste the link on your browser and try the prototype!


