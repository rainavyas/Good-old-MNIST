import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import csv
from models import Simple_DNN

IMG_DIM = 28

# Set seed for reproducibility
seed = 1
torch.manual_seed(seed)

# Read in Data
y = []
X = []

with open('train.csv', newline='') as csvfile:
    myreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    counter = 0

    for row in myreader:
        # ignore header of file
        if counter == 0:
            counter +=1
            continue
        row_vals = row[0].split(',')
        row_vals = [int(val) for val in row_vals]
        X.append(row_vals[1:])
        y.append(row_vals[0])

# Separate into Training and Dev sets (80:20 ratio)
training_size = int(0.8*len(X))
X_train = X[:training_size]
X_dev = X[training_size:]
y_train = y[:training_size]
y_dev = y[training_size:]


# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_dev = torch.FloatTensor(X_dev)
y_train = torch.LongTensor(y_train)
#y_dev = torch.FloatTensor(y_dev)

# Reshape input tensors into images
X_train = torch.reshape(X_train, (-1, IMG_DIM, IMG_DIM))
X_dev = torch.reshape(X_train, (-1, IMG_DIM, IMG_DIM))


# Mini-batch size
bs = 100
epochs = 80
lr = 1e-2

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X_train, y_train)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

# Construct model
my_model = Simple_DNN(IMG_DIM, 10)
my_model = my_model.float()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)

# pass through model
for epoch in range(epochs):
    my_model.train()

    for xb, yb in train_dl:
        y_pred = my_model.forward(xb)
        # Compute cross entropy loss
        loss = criterion(y_pred, yb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("loss:", loss.item())
    # Report accuracy on the dev set
    y_dev_pred_likelihoods = my_model.forward(X_dev)
    y_dev_pred = torch.argmax(y_dev_pred_likelihoods, dim = 1)
    y_dev_pred_list = y_dev_pred.tolist()
    total = 0
    correct = 0
    for real, pred in zip(y_dev, y_dev_pred_list):
        total+=1
        diff = real - pred
        if abs(diff) < 0.1:
            correct +=1
    acc = correct/total

    print("Epoch: ", epoch, "Dev Accuracy: ", acc)
