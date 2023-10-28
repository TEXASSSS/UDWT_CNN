import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob

import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import statistics


# Define the dataset class
def Normalization(Signal):
    mean = np.mean(Signal)
    std = np.std(Signal)
    Norm_Signal = (Signal - mean) / std
    return Norm_Signal


class DatasetTrain(Dataset):
    def __init__(self, root):
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        # sort file names
        self.input_paths = sorted(
            glob(os.path.join(self.root, '{}/*_data.npy'.format("Train_data"))))
        self.label_paths = sorted(
            glob(os.path.join(self.root, '{}/*_lab.npy'.format("Train_lab"))))
        self.name = os.path.basename(root)

        # print(self.input_paths)
        # print(self.label_paths)

        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No signal/labels are found in {}".format(self.root))

    def __getitem__(self, index):

        Signal = np.load(self.input_paths[index])
        Signal = Normalization(Signal)

        Label = np.load(self.label_paths[index])
        # print(self.input_paths[index], " is loaded")
        # print(type(Signal), Signal.shape)

        return Signal, Label

    def __len__(self):
        return len(self.input_paths)


class DatasetTest(Dataset):
    def __init__(self, root):
        self.root = root

        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        # sort file names
        self.input_paths = sorted(
            glob(os.path.join(self.root, '{}/*_data.npy'.format("Test_data"))))
        self.label_paths = sorted(
            glob(os.path.join(self.root, '{}/*_lab.npy'.format("Test_lab"))))
        self.name = os.path.basename(root)

        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No signal/labels are found in {}".format(self.root))

    def __getitem__(self, index):

        Signal = np.load(self.input_paths[index])
        Signal = Normalization(Signal)

        Label = np.load(self.label_paths[index])
        return Signal, Label

    def __len__(self):
        return len(self.input_paths)


# Define the CNN architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.AvgPool2d((9, 2))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(7296, 512*16),
            nn.ReLU(),
            nn.Linear(512*16, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # print("Initial shape:", x.shape)
        x = self.conv(x)
        # print("After conv layer", x.shape)
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor before the fully connected layers
        # print("After flatten layer", x.shape)
        x = self.classifier(x)
        # print("After fully-connected layer", x.shape)
        return x


# Set the data directories
# train_data_dir = os.path.abspath("./underwaterData/Train_data/")
# train_label_dir = os.path.abspath("./underwaterData/Train_lab/")
# test_data_dir = os.path.abspath("./underwaterData/Test_data/")
# test_label_dir = os.path.abspath("./underwaterData/Test_lab/")
dirct = r"C:\Users\MOJITO\Desktop\passion\S&R\UdWt\潜水器数据集\Numpy_data\Data"

# Create dataset and dataloaders
train_dataset = DatasetTrain(dirct)
test_dataset = DatasetTest(dirct)

# Initialize the model
model = CNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000422) # lr=0.0009999, betas=(0.99, 0.999)
criterion = nn.CrossEntropyLoss()
num_epochs = 30
NUM_CLASSES = 5  # Replace with the actual number of classes in your dataset
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)


# Define loss function and optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0018)
# criterion = nn.CrossEntropyLoss()

# Training loop
# def train(model, num_epochs, learning_rate):

for epoch in range(num_epochs):
    # model = CNNModel(NUM_CLASSES).to("cpu")
    pbar = tqdm(train_loader, desc="epoch: " + str(epoch))
    for index, (inputs, labels) in enumerate(pbar):
        # inputs, labels = inputs.to("cpu"), labels.to("cpu")
        # print(inputs.shape, inputs)
        outputs = model(inputs)
        # outputs, _ = torch.max(outputs, 1)
        labels = torch.squeeze(labels, 1)
        labels = labels.long()
        # print("Output shape:", outputs)
        # print("Target shape:", labels.shape, labels)
        # labels = labels.float()
        loss = criterion(outputs, labels)
        # print("Loss = ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({
            'loss': loss.item()
        })
        # outputs = torch.squeeze(outputs)
        # labels = torch.Tensor(np.int64(labels))
        # print("pred is: ", outputs)
        # print("Input shape:", inputs.shape)
print("Training finished")

# Evaluation on the test set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # inputs, labels = inputs.to("cpu"), labels.to("cpu")
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = torch.squeeze(labels, 1)
        # outputs = torch.squeeze(outputs, 1)
        # labels = torch.squeeze(labels, 1)
        # labels = torch.LongTensor(np.int64(labels))
        # print("shape of outputs: ", outputs.shape)
        # print("shape of labels: ", labels.shape)
        # print("Prediction result is:", preds, "Actual label is: ", labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
print(len(all_preds))
print(len(all_labels))
print(all_preds)
print(all_labels)
accuracy = accuracy_score(all_labels, all_preds)
print("Test accuracy = ", accuracy)
