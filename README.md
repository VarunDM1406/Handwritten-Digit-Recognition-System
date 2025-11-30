📌 Overview
This project implements a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN) trained on digit images.
The system processes an input image of a handwritten digit (0–9) and predicts the correct digit with high accuracy.
This is a classic computer vision project widely used in banking, postal mail sorting, document processing, and automated evaluation systems.

🚀 Features
Recognizes handwritten digits (0–9)

Uses a lightweight yet powerful CNN architecture

High accuracy on test data

Preprocessing of images for better prediction

Easy to run and modify

Model saved as .h5 file for reuse

🧠 Model Architecture

The CNN model includes:

Conv2D Layer – Extracts visual features

MaxPooling2D – Reduces spatial dimensions

Flatten Layer – Converts 2D features into 1D

Dropout Layer – Prevents overfitting

Dense Layer (Softmax) – Outputs probability of each digit (0–9)

📂 Project Structure
Handwritten-Digit-Recognition-System/
│
├── finalcode.py                 # Main python script
├── mnist_cnn_model.h5           # Trained model file
└── README.md                    # Project documentation

📦 Requirements

Install the required libraries:

pip install tensorflow
pip install numpy
pip install matplotlib
pip install opencv-python

▶️ How to Run

Clone the repository:

git clone https://github.com/VarunDM1406/Handwritten-Digit-Recognition-System.git


Navigate into the project folder:

cd Handwritten-Digit-Recognition-System


Run the Python script:

python finalcode.py

📊 Dataset

The model is trained on the MNIST dataset, consisting of:

70,000 images of handwritten digits

28×28 grayscale images

10 classes (0–9)

📈 Results

Achieved high classification accuracy on test images

Successfully recognizes custom inputs

Model generalizes well to different handwriting styles

Digit	Example Prediction
5	✔ Correct
8	✔ Correct
3	✔ Correct

(Images can be added if you want — I can generate the markdown for them.)

🔮 Future Enhancements

Deploy the model using a web interface

Add support for handwritten alphabets (A–Z)

Use advanced CNN models like ResNet or MobileNet

Implement real-time digit capture from camera

📜 License

This project is open-source and free to use for learning and academic purposes.

👤 Author

Varun Dev Mittal (GitHub: VarunDM1406)
Feel free to contact or fork the project!
