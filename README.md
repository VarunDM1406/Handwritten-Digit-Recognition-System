1. ABSTRACT
Handwritten digit recognition is an essential task in pattern recognition and machine learning, widely used in postal automation, banking systems, academic evaluation, and document processing.
 This project presents a Convolutional Neural Network (CNN)-based model capable of classifying handwritten digits (0–9) with high accuracy. The system preprocesses the input image, extracts features using convolutional layers, and predicts the digit using fully connected layers.
 The model demonstrates strong performance and generalization, making it suitable for real-world applications involving handwritten data.
2. INTRODUCTION
Handwritten digit recognition is a classic problem in computer vision and machine learning. Traditional methods relied on manual feature extraction, but recent advances in deep learning—especially CNNs—allow systems to learn features directly from raw pixel data.
This project aims to develop a robust digit recognition model using CNNs. The system takes a grayscale image of a handwritten digit, processes it through multiple neural network layers, and classifies it into one of the ten categories (0–9).
The project also includes dataset preprocessing, model training, evaluation, and real-time testing using custom images.
3. OBJECTIVES OF THE PROJECT
The main objectives of this project are:
To implement a CNN-based model for handwritten digit classification
To preprocess images for better feature extraction
To train and evaluate the model on digit datasets
To test the model on real-world handwritten samples
To demonstrate practical applications of digit recognition
4. SYSTEM REQUIREMENTS
Software Requirements
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Jupyter Notebook / VS Code
Hardware Requirements
Minimum 4GB RAM
GPU recommended (optional but speeds up training)
5. DATASET DESCRIPTION
The dataset consists of thousands of handwritten digit images (0–9). Each image is:
28×28 pixels
Grayscale
Centered and normalized
The dataset is divided into:
Training set – used to teach the model
Validation set – used during training to avoid overfitting
Test set – used to evaluate the model’s final performance
6. METHODOLOGY
The project follows the typical machine learning workflow:
6.1 Data Preprocessing
Image resizing
Grayscale normalization
Reshaping to match CNN input
One-hot encoding of labels
6.2 Model Building
A Sequential CNN model was implemented with the following layers:
Conv2D Layer – extracts features such as edges and curves
MaxPooling2D Layer – reduces spatial dimensions
Flatten Layer – converts 2D feature maps to 1D
Dropout Layer – reduces overfitting
Dense Output Layer – classifies input into 10 categories
6.3 Model Training
Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Epochs: 10–20
Batch Size: 32 / 64
6.4 Model Evaluation
Metrics used:
Accuracy
Loss
Confusion Matrix
7. MODEL ARCHITECTURE
The final CNN model includes:
Conv2D(32 filters, 3×3 kernel, ReLU activation)
MaxPooling2D(2×2)
Flatten Layer
Dropout (0.25)
Dense(10 units, softmax activation)
This architecture is lightweight, efficient, and performs well on digit recognition tasks.
8. RESULTS AND DISCUSSION
The model achieved high accuracy on the test set, demonstrating strong ability to recognize handwritten digits. Key observations:
The CNN model learns distinct patterns such as curves (digit 3), loops (digit 8), and straight lines (digit 1).
Dropout significantly improved generalization by preventing overfitting.
Preprocessing steps helped the model handle variations in handwriting style.
Real-world digit images were also tested, and the model was able to correctly classify most samples.
9. APPLICATIONS
Handwritten digit recognition is used in:
Postal mail sorting
Bank cheque processing
Automatic form reading
Attendance systems
Exam evaluation
Mobile number pad recognition
Document digitization
10. CONCLUSION
This project successfully implemented a Convolutional Neural Network capable of recognizing handwritten digits with high accuracy. The system is efficient, scalable, and can be extended for more complex handwritten character recognition tasks.
Future improvements may include:
Using more advanced CNN architectures
Adding data augmentation
Deploying the model using web or mobile interfaces
Expanding recognition to alphabets or full words
11. REFERENCES
Yann LeCun et al. “Gradient-Based Learning Applied to Document Recognition.”
MNIST Database, http://yann.lecun.com/exdb/mnist/
TensorFlow Documentation – https://www.tensorflow.org
Keras Documentation – https://keras.io
