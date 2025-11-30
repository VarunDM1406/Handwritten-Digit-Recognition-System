import numpy as np 
import matplotlib.pyplot as plt 

import keras 

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test)=mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
