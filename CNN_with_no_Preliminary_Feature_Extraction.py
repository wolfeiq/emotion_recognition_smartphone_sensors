import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# Best parameters from Optuna optimization for 3 emotions (relaxed, happy, frustrated)

CONFIG = {
    "input_channels": 3,
    "num_classes": 3,
    "lr": 0.0009017984783496502,  
    "batch_size": 64,  
    "num_epochs": 50,
    "dropout_rate": 0.4781630617226421,  
    "kernel_size": 5,  
    "num_filters": [32, 128, 64],  
    "window_size": 500,
    "overlap": 0.5,
    "optimizer": "adam",  
    "weight_decay": 1.1463437865528338e-06,  
    "scheduler_patience": 6,  
    "scheduler_factor": 0.26977887803191647, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class Smartphone(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data.transpose(0, 2, 1))
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_from_excel(file):
    emotions = ['chill', 'frustrated', 'happy']
    accelerometer, labels = [], []

    for label, emotion in enumerate(emotions):
        df = pd.read_excel(file, sheet_name=emotion)
        xyz = df[['x', 'y', 'z']].values
        windows = preprocess(xyz, window_size=500, overlap=0.5)

        emotion_labels = [label] * len(windows)
        accelerometer.append(windows)
        labels.extend(emotion_labels)

    accelerometer = np.concatenate(accelerometer, axis=0)
    labels = np.array(labels)

    return accelerometer, labels


def preprocess(sensor_data, window_size=500, overlap=0.5):
    stride = int(window_size * (1 - overlap))
    n_samples = len(sensor_data)
    n_windows = ((n_samples - window_size) // stride) + 1
    
    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        if end <= n_samples:
            windows.append(sensor_data[start:end])
    
    return np.array(windows)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        num_filters = config["num_filters"]
        kernel_size = config["kernel_size"]
        dropout_rate = config["dropout_rate"]
        input_channels = config["input_channels"]
        num_classes = config["num_classes"]
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, num_filters[0], kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_filters[1], num_filters[2], kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_filters[2], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(config, model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config["lr"], 
            weight_decay=config["weight_decay"] 
        )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=config["scheduler_patience"], factor=config["scheduler_factor"]
    )
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()


        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        scheduler.step(val_loss)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'cnn_er_hyperparameter_optimized.pth')


def main():
    accelerometer_file = '...path...'
    accelerometer_windows, labels = load_from_excel(accelerometer_file)

    X_train, X_val, y_train, y_val = train_test_split(accelerometer_windows, labels, test_size=0.2, random_state=42)
    
    train_dataset = Smartphone(X_train, y_train)
    val_dataset = Smartphone(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = CNN(CONFIG).to(CONFIG["device"])
    
    train(CONFIG, model, train_loader, val_loader)



if __name__ == "__main__":
    main()