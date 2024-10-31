# Digit Recoginition
This project is a deep learning-based digit recognition model using the MNIST dataset. It classifies handwritten digits (0-9) and can predict digits from custom images. The model is built using TensorFlow and Keras and trained on the MNIST dataset.

## Overview
The MNIST dataset is a widely used benchmark for handwritten digit classification, containing 60,000 training images and 10,000 test images, each labeled from 0 to 9. This project demonstrates:

Loading and visualizing data from the MNIST dataset.
Building, training, and evaluating a neural network model for digit recognition.
Using a trained model to predict digits from new, user-provided images.

## Tools & Libraries
**Numpy**: For numerical operations and data manipulation.

**Matplotlib & Seaborn**: For data visualization and plotting the confusion matrix.

**OpenCV**: For image processing and custom image input handling.

**PIL (Python Imaging Library)**: To manage image file formats.

**TensorFlow & Keras**: For building and training the neural network model.

## Project Structure
**Load and Explore Data**: Loads the MNIST dataset and explores the structure of training and testing data.

**Data Preprocessing**: Scales pixel values for better model performance.

**Model Building**: Creates a neural network using Keras with three layers:

Flatten layer to convert 2D images into 1D arrays.

Two dense layers with ReLU activation.

Output layer with a sigmoid activation for 10 classes (0-9).

**Model Training and Evaluation**: Compiles the model with Adam optimizer, trains it on the training dataset, and evaluates its accuracy on the test dataset.

**Prediction on Custom Images**: Allows users to upload custom images of digits to be predicted by the trained model.

## Code Walkthrough
#### Data Loading and Visualization
Loads the MNIST dataset and displays sample images along with their labels to understand the data distribution.

##### Data Preprocessing
Scales pixel values to a range of 0-1 by dividing by 255.

##### Model Architecture

**Flatten**: Converts each 28x28 image into a 1D array of 784 values.

**Dense Layers**:
Two hidden layers with 50 neurons each and ReLU activation.

Output layer with 10 neurons and sigmoid activation for multi-class classification.

#### Model Compilation and Training

Compiles the model using sparse_categorical_crossentropy loss and trains it over 10 epochs.

#### Evaluation and Confusion Matrix
Evaluates model accuracy on the test set and generates a confusion matrix to visualize performance.

#### Custom Image Prediction
Allows users to provide the path to an image file containing a digit. The image is preprocessed and reshaped to the input format (28x28) and passed through the model to predict the digit.

## How to Use
Clone the repository and navigate to the project directory.

Run the Jupyter notebook or Python file to train the model.

Upload a custom image to predict the digit by providing the image path as input.

## Results
Displays the model’s accuracy and a confusion matrix using Seaborn's heatmap, showing the correct vs. incorrect predictions across classes.

## Acknowledgments
This project uses the MNIST dataset, a classic dataset in machine learning. Special thanks to TensorFlow and Keras for model-building support.

## License
MIT License
