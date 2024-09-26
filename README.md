# Handwritten Digit Recognition
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. It serves as an example of how deep learning can be applied to image classification tasks and can also predict digits from user-provided images.

## 📚 Overview
The MNIST dataset is a classic dataset used for training various image processing systems. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9) in grayscale, each represented as a 28x28 pixel image.

The objective of this project is to build and train a CNN model to recognize these digits with high accuracy and to demonstrate the model's ability to generalize to unseen data, including your custom handwritten images.

## 🧠 Model Architecture
The Convolutional Neural Network (CNN) used in this project has the following architecture:

1. **Conv2D Layer**: 32 filters, kernel size (3x3), ReLU activation
2. **MaxPooling2D Layer**: Pool size (2x2)
3. **Conv2D Layer**: 64 filters, kernel size (3x3), ReLU activation
4. **MaxPooling2D Layer**: Pool size (2x2)
5. **Flatten Layer**: Converts the matrix into a 1D vector
6. **Dense Layer**: 64 units, ReLU activation
7. **Output Dense Layer**: 10 units, softmax activation for multi-class classification

The model is compiled using the `Adam` optimizer, `categorical_crossentropy` loss function, and `accuracy` as the performance metric.

## 🚀 Getting Started

### Prerequisites
Before running this project, make sure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow (for custom image handling)

Install the required packages with:
```bash
pip install tensorflow keras numpy matplotlib pillow
```

### Clone the Repository
```bash
git clone https://github.com/username/handwritten-digit-classification.git
cd handwritten-digit-classification
```

### Training the Model
To train the model using the MNIST dataset, run:
```bash
python train_model.py
```
The script will automatically download the dataset, preprocess it, and train the CNN model. Training progress will be displayed, including loss and accuracy for both training and validation data.

### Evaluating the Model
After training, the model is evaluated on the test data to measure its accuracy. You can run:
```bash
python evaluate_model.py
```

### Predicting Custom Images
You can test the model with your own handwritten digit images:
1. Create or upload a grayscale image of a digit (28x28 pixels) in `.png` format.
2. Place the image in the project directory and specify the path in `predict_image.py`.
3. Run the prediction script:
   ```bash
   python predict_image.py
   ```

The model will output the predicted digit along with the confidence scores.

## 📊 Results
The model achieves high accuracy on the MNIST test dataset. You can visualize training history and predictions with the provided scripts, which include a plot of the training/validation accuracy and examples of predicted test images.

## 📂 Project Structure
- `train_model.py`: Main script to train the CNN model.
- `evaluate_model.py`: Script to evaluate the model on the test set.
- `predict_image.py`: Script to predict custom digit images.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies for easy setup.

## 🔧 Technologies Used
- **TensorFlow & Keras**: For building and training the CNN model
- **NumPy**: For numerical operations
- **Matplotlib**: For plotting and visualizing results
- **Pillow (PIL)**: For image processing

## 🤝 Contributing
Contributions are welcome! If you'd like to enhance the model, improve documentation, or add features, feel free to open an issue or submit a pull request.


