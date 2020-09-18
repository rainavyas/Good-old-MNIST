# Good-old-MNIST
Playing around with computer vision

## Task
Classify 28 by 28 dimension images of hand-drawn digits 0-9.

## Approach

### Simple DNN
Two layer deep neural network with input vector dimension 28^2 and output 10 dimension (unnormalised probability of each class). Trained using cross entropy. Achieves ~96% accuracy.

### Simple CNN
Single convolutional layer (with 8 output channels, stride = 1, padding =1, kernel_size = 3) on 28 by 28 input image of a single channel. Followed by max-pooling and then ouput converted to a single feature vector, passsed through two fully connected layers giving a 10 dimensional output (unnormalised probabilities). Achieves ~97.5% accuracy.

### Less Simple CNN
X->convolution->ReLU->MaxPool->convolution->ReLU->MaxPool->FC->ReLU->FC->unnormalised class probabilities. Achieves 98.3% accuracy.

### Simple CNN Ensemble
The above model was trained with 5 different seeds and the output probabilities averaged during evaluation. Achieves 98.2% accuracy.

### Less Simple CNN Ensemble
The above model was trained with 5 different seeds and the output probabilities averaged during evaluation. Achieves 98.9% accuracy.
