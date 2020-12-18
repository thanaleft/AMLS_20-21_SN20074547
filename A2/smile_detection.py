import face_recognition as fr
import cv2
import numpy as np
import os
import dlib  
import numpy as np
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
import random
#---------------------------Haar Cascade Method----------------------------
def CV_smile(k=70):
    smile_cascade = cv2.CascadeClassifier("./A2/haarcascade_smile.xml")
    base_dir = "./Datasets"
    file_path = os.path.join(base_dir, "celeba_test/labels.csv")
    image_dir = os.path.join(base_dir, "celeba_test/img")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    smile_labels = []
    pred_smile = []
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir, line.split('\t')[1])
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smile_labels.append(int(line.split('\t')[3].replace('\n', '')))
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor= 1.16,
            minNeighbors=k,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(smiles) > 0:
            pred_smile.append(1)
        else:
            pred_smile.append(-1)
    smile_labels = np.array(smile_labels)
    pred_smile = np.array(pred_smile)
    print("Classification report for classifier:\n%s\n"
      % (classification_report(smile_labels, pred_smile)))

#--------------------------------Haar + CNN method--------------------
#data pre-processing
def data_augment(img):
    rows, cols = img.shape
    lst = ['move', 'rot', 'flip']
    op = str(random.choice(lst))
    if op == 'move':
        x = random.randint(4,11)
        y = random.randint(4,11)
        M = np.float32([[1,0,x],[0,1,y]])
        dst = cv2.warpAffine(img, M, (cols,rows))
        return dst
    elif op == 'rot':
        angle = random.randint(6, 15)
        s = random.choice([-1,1])
        M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0),s*angle,1)
        dst = cv2.warpAffine(img, M, (cols,rows))
        return dst
    elif op == 'flip':
        dst = cv2.flip(img, 1)
        return dst

def make_dataset():
    #Make training set
    print("Making training set:")
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "celeba")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    haarcascade_face_dir = "A2"
    haarcascade_face_dir = os.path.join(haarcascade_face_dir, "haarcascade_frontalface_default.xml")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    detector = cv2.CascadeClassifier(haarcascade_face_dir)
    trainX = []
    trainY = []
    failnum = 0
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir, line.split('\t')[1])
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(28, 28),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) != 1:
            failnum += 1
            continue
        for (fX, fY, fW, fH) in rects:
            roi = gray[fY:fY + fH, fX:fX + fW]  
            roi = cv2.resize(roi, (96, 96))
            roi_aug = data_augment(roi)
        roi = np.expand_dims(roi, axis = 0)
        roi_aug = np.expand_dims(roi_aug, axis = 0)
        trainX.append(roi)
        trainX.append(roi_aug)
        if int(line.split('\t')[3].replace('\n', '')) == -1:
            trainY.append(0)
            trainY.append(0)
        else:
            trainY.append(1)
            trainY.append(1)
    trainX = np.array(trainX, dtype= "float") / 255.0 
    trainY = np.array(trainY)
    

    #Make training set
    print("Making test set:")
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "celeba_test")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    haarcascade_face_dir = "A2"
    haarcascade_face_dir = os.path.join(haarcascade_face_dir, "haarcascade_frontalface_default.xml")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    detector = cv2.CascadeClassifier(haarcascade_face_dir)
    testX = []
    testY = []
    failnum = 0
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir, line.split('\t')[1])
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(28, 28),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) != 1:
            failnum += 1
            continue
        for (fX, fY, fW, fH) in rects:
            roi = gray[fY:fY + fH, fX:fX + fW]  
            roi = cv2.resize(roi, (96, 96))
        roi = np.expand_dims(roi, axis = 0)
        testX.append(roi)
        if int(line.split('\t')[3].replace('\n', '')) == -1:
            testY.append(0)
        else:
            testY.append(1)
    testX = np.array(testX, dtype= "float") / 255.0 
    testY = np.array(testY)
    
    trainX = torch.from_numpy(trainX)
    testX = torch.from_numpy(testX)
    trainY = torch.from_numpy(trainY)
    testY = torch.from_numpy(testY)
    
    return trainX, testX, trainY, testY

#Network part
# Define network structure
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.features = nn.Sequential(#input 1*96*96
            nn.Conv2d(1, 32, 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 128, 3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(#input 64*52*42
            nn.Dropout(0.5),
            nn.Linear(36864, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        #print(x.size())
        x = self.classifier(x)
        return x

def train_CNN(TrainX, TrainY):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyper parameter setting
    EPOCH = 200
    BATCH_SIZE = 128
    LR = 0.005

    # Define the data processing method
    transform = transforms.ToTensor()

    # Define loss function and optimizer
    model = Network().to(device)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0, std=0.02)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

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
            scheduler.step()

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
    model_path = "A2"
    model_path = os.path.join(model_path, 'Haar_CNN_model_%03d.pth' % (epoch + 1))
    torch.save(model.state_dict(), model_path)
    if device == 'cuda':
        torch.cuda.empty_cache()

def test_CNN(TestX, TestY, path='./A2/Haar+CNN.pth'):
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
    