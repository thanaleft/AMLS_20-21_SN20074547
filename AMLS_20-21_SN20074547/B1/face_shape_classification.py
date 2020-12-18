import face_recognition as fr
import cv2
import numpy as np
import os
import dlib  
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse

#---------------------------SVM----------------------
def make_dataset_svm():
    #Makeing training set
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "cartoon_set")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    trainX = []
    trainY = []
    fail_num = 0
    for line in tqdm(lines):
        image_path = os.path.join(image_dir, line.split('\t')[3].replace('\n', ''))
        print(image_path)
        try:
            img = cv2.imread(image_path)
            face_embedding = fr.face_encodings(img)
            if len(face_embedding) != 1:
                fail_num += 1
                continue
            trainX.append(face_embedding)
            trainY.append(int(line.split('\t')[2]))
        except Exception as e:
            print(e)
            print(image_path)
            continue
            
    trainX = np.array(trainX).squeeze()
    trainY = np.array(trainY)

    #Making test set
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "cartoon_set_test")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    testX = []
    testY = []
    fail_num = 0
    for line in tqdm(lines):
        image_path = os.path.join(image_dir, line.split('\t')[3].replace('\n', ''))
        print(image_path)
        try:
            img = cv2.imread(image_path)
            face_embedding = fr.face_encodings(img)
            if len(face_embedding) != 1:
                fail_num += 1
                continue
            testX.append(face_embedding)
            testY.append(int(line.split('\t')[2]))
        except Exception as e:
            print(e)
            print(image_path)
            continue
            
    testX = np.array(testX).squeeze()
    testY = np.array(testY)
    
    return trainX, testX, trainY, testY

def Grid_SVM(trainX, trainY, testX):
    param_grid = {'C': [1e1, 1e2],
              'gamma': [10, 100], 
              'decision_function_shape':["ovo"],
                "degree":[3,4]}
    model = GridSearchCV(svm.SVC(kernel='rbf', class_weight="balanced", verbose=True),
            param_grid, cv=5)
    model.fit(trainX, trainY)
    print(model.best_params_, model.best_score_)
    y_pred = model.best_estimator_.predict(testX)
    return y_pred

#--------------------------CNN-----------------------

def make_dataset():
    #Makeing training set
    print("Making training set:")
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "cartoon_set")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    trainX = []
    trainY = []
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir, line.split('\t')[3].replace('\n', ''))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200,200))
        img = np.expand_dims(img, axis = 0)
        img = img / 255.0
        trainY.append(int(line.split('\t')[2]))
        trainX.append(img)
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    #Makeing test set
    print("Making test set:")
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "cartoon_set_test")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    testX = []
    testY = []
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir, line.split('\t')[3].replace('\n', ''))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200,200))
        img = np.expand_dims(img, axis = 0)
        img = img / 255.0
        testY.append(int(line.split('\t')[2]))
        testX.append(img)
    testX = np.array(testX)
    testY = np.array(testY)
    
    trainX = torch.from_numpy(trainX)
    testX = torch.from_numpy(testX)
    trainY = torch.from_numpy(trainY)
    testY = torch.from_numpy(testY)
    
    return trainX, testX, trainY, testY

# Define network structure
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, 5, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
        )

        self.classifier = nn.Sequential(#input 64*52*42
            nn.Dropout(0.4),
            nn.Linear(147456, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5),
            nn.Softmax(dim=1)
        )

    # Define forward propagation
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        #print(x.size())
        x = self.classifier(x)
        return x

def train_CNN(TrainX, TrainY):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyper parameter setting
    EPOCH = 32
    BATCH_SIZE = 64
    LR = 0.0001

    # Define the data processing method
    transform = transforms.ToTensor()

    # Define loss function and optimizer
    model = Network().to(device)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0, std=0.02)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.002)

    # Make data loader
    train_set = torch.utils.data.TensorDataset(TrainX, TrainY)
    l = len(train_set)
    train_set, val_set = torch.utils.data.random_split(train_set, [int(0.8*l), l-int(0.8*l)])
    trainloader = torch.utils.data.DataLoader(train_set,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
    valloader = torch.utils.data.DataLoader(val_set,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
    # Training
    train_loss = []
    val_loss = []
    print("Start training:")
    for epoch in range(EPOCH):
        sum_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # forward + backward
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            
        print('Train Epoch: {} \tAverage Loss: {:.6f}'.format(epoch, sum_loss / len(trainloader)))
        train_loss.append(sum_loss / len(trainloader))
        # validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            valloss = 0
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                images = images.float()
                labels = labels.long()
                outputs = model(images)
                valloss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                valloss / len(valloader), correct, len(valloader.dataset),
                100. * correct / len(valloader.dataset)))
            val_loss.append(valloss / len(valloader))
    model_path = "B1"
    model_path = os.path.join(model_path, "face_shape_CNN_%03d.pth"% (epoch + 1))
    torch.save(model.state_dict(), model_path)
    if device == 'cuda':
        torch.cuda.empty_cache()

def test_CNN(TestX, TestY, path='B1/face_shape_CNN.pth'):
    print("Start testing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = torch.utils.data.TensorDataset(TestX, TestY)
    testloader = torch.utils.data.DataLoader(test_set,
                                         batch_size=64,
                                         shuffle=True)
    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    testloss = 0
    correct = 0
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        images = images.float()
        labels = labels.long()
        outputs = model(images)
        testloss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
    acc = 100. * correct / len(testloader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        testloss / len(testloader), correct, len(testloader.dataset),
        acc))
    return acc