import torch


class Basic_two_layer(torch.nn.Module):
    # This doesn't train well
    def __init__(self, image_size, num_digits):

        super(Basic_two_layer, self).__init__()
        self.layer1 = torch.nn.Linear(image_size, num_digits)
        self.layer2 = torch.nn.Linear(image_size, 1)

    def forward(self, X):
        out1= self.layer1(X)
        relu = torch.nn.ReLU()
        out1_act = relu(out1)
        #out1_act = torch.sigmoid(out1)
        y_unnormal = self.layer2(torch.transpose(out1_act, 1,2))
        y_out = y_unnormal.squeeze()
        return y_out

# Make new simpler model that converts to a single 28^2 vector and passes through layers
class Simple_DNN(torch.nn.Module):
    def __init__(self, image_size, num_digits):
        super(Simple_DNN, self).__init__()
        self.layer1 = torch.nn.Linear(image_size**2, 200)
        self.layer2 = torch.nn.Linear(200, num_digits)
        self.image_size = image_size

    def forward(self, X):
        X_1d = torch.reshape(X, (-1, self.image_size**2))
        X_1d_squeezed = X_1d.squeeze()
        h1 = self.layer1(X_1d_squeezed)
        relu = torch.nn.ReLU()
        h1_nonlinear = relu(h1)
        h2 = self.layer2(h1_nonlinear)
        # Return unnormalised probabilities as loss function of cross entropy includes softmax
        return h2

class Simple_CNN(torch.nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # image dimension is 28, stride =2 => 28/2=14
        self.fc1 = torch.nn.Linear(8*14*14, 50)
        # 10 output classes
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, X):
        # Reshape to have single channel
        x = X.unsqueeze(1)
        x_conv1 = self.conv1(x)
        relu = torch.nn.ReLU()
        x_act = relu(x_conv1)
        x_pooled = self.pool(x_act)
        x_pooled_reshaped = x_pooled.reshape(-1, 8*14*14)
        x_fc1 = self.fc1(x_pooled_reshaped)
        x_fc1_act = relu(x_fc1)
        y = self.fc2(x_fc1_act)
        return y
