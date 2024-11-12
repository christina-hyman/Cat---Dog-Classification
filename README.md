# CS 672 (CRN: 74071): Introduction to Deep Learning
# Christina Hyman
## Project Overview
In this project, I applied transfer learning using the pre-trained InceptionV3 model in TensorFlow for a binary classification task on a dataset containing 25,000 cat and dog images. The goal was to classify the images as either a cat or a dog, using both a base CNN model built with TensorFlow.Keras and a transfer learning approach with InceptionV3.

### Key Results:
**InceptionV3 Model**: Achieved a validation accuracy of 99.08% after 9 epochs, significantly outperforming the base CNN model.
**TensorFlow.Keras Base Model**: Achieved a validation accuracy of 76.18% after the same number of epochs.

## Dataset
The dataset contains 25,000 images of cats and dogs, with 12,499 images for each class. These images are initially downloaded from a Microsoft server as a zip file and extracted to the /tmp directory in Google Colab.

### Data Preprocessing:
1. Corrupted Image Removal: The `is_image_corrupted` function checks for and flags any corrupted or unreadable images using the PIL library. The remove_corrupted_images function then removes those corrupted images from the dataset.
2. Dataset Split: The dataset is split into training and validation sets using an 80-20 split ratio. The `split_data` function organizes the images into "cat" and "dog" subfolders within the train and validation directories.
3. Image Augmentation: For the training set, data augmentation is applied to enhance generalization. This includes random rotations, shifts, shears, zooms, and flips. For validation, only rescaling is applied.

### Image Generators:
`train_datagen`: Rescales images and applies random transformations for data augmentation.
`validation_datagen`: Only rescales images. Both generators load and preprocess the images in batches of 32, resizing them to 224x224 pixels.

## Model Implementation & Evaluation

### TensorFlow.Keras Base Model:
**Model Architecture**:
The base model consists of two convolutional layers followed by pooling layers to extract hierarchical features. The output is flattened and passed through dense layers with a sigmoid activation function for binary classification.
**Training & Evaluation**:
The model is trained for 10 epochs using the Adam optimizer with a learning rate of 0.001 and binary cross-entropy loss.
After training, the model achieved a validation accuracy of 87.50%. However, the fluctuating training history suggested potential instability and overfitting.
**Run time**: 17 minutes.

### Transfer Learning using InceptionV3:
**Model Architecture**:
The InceptionV3 model is pre-trained on ImageNet and fine-tuned for this binary classification task. The early layers are frozen, and custom layers are added for the final classification.
**Training & Evaluation**:
* The InceptionV3 model was trained using the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.
* The model achieved a validation accuracy of **99.08%** after 9 epochs, which is a significant improvement over the base model.
**Run time**: 25 minutes 14 seconds.

## Conclusion

**TensorFlow Base Model**:
* Shows signs of overfitting and instability in validation accuracy, suggesting that the model may benefit from more regularization or additional data augmentation. Possible improvements include dropout layers, L2 regularization, or early stopping.

**InceptionV3 Model**:
* Demonstrates superior performance, suggesting that the pre-trained model generalizes well to this classification task. The current model appears to be well-optimized.

### Future Recommendations:
**For the TensorFlow Base Model**:
* Explore regularization techniques to address overfitting.
* Consider adding more data augmentation to improve model robustness.
* Simplify the model if overfitting persists.

**For the InceptionV3 Model**:
* The model is already performing well, but additional fine-tuning or further experiments could be considered for even higher accuracy.

Overall, the **InceptionV3 model** outperforms the base model and is the better choice for this binary image classification task.

## Requirements:
* TensorFlow
* PIL (Python Imaging Library)
* NumPy
* Matplotlib (for visualizing training progress and image transformations)
