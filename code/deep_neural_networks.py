import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

class Net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[1]),
            nn.Flatten(),
            nn.Linear(input_shape[1]*input_shape[2]*input_shape[3]*input_shape[4], 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(250, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(inplace=True),
            nn.Linear(25, output_shape),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def algorithm_fcnn(x_train, x_val, y_train, y_val, log_write):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available: ", torch.cuda.get_device_name(device=None))
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU")

    train_dataset = TensorDataset(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(x_val).to(device), torch.Tensor(y_val).to(device))
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=True)

    net = Net(input_shape=x_train.shape, output_shape=len(np.unique(y_train))).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss()

    if log_write:
        writer = SummaryWriter()

    for epoch in range(50):
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        train_loss /= len(train_loader)

        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                val_loss += criterion(outputs, labels.long()).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        if log_write:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        print("Epoch: {}/50, Training Loss: {:.4f}, Training Accuracy: {:.2f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%".format(
            epoch + 1,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if log_write:
        writer.close()

    return net