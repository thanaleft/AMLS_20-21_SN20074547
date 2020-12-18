import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

def make_dataset():
    #Make training set
    print("Making training set:")
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "cartoon_set")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    haarcascade_eye_dir = "B2"
    haarcascade_eye_dir = os.path.join(haarcascade_eye_dir, "haarcascade_eye.xml")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    detector = cv2.CascadeClassifier(haarcascade_eye_dir)
    trainX = []
    trainY = []
    failnum = 0
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir, line.split('\t')[3].replace('\n', ''))
        img = cv2.imread(image_path)
        rects = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(28, 28),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            failnum += 1
            continue
        for (fX, fY, fW, fH) in rects:
            roi = img[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (50, 50))
            trainX.append(roi)
            break
        trainY.append(int(line.split('\t')[1]))
    trainX = np.array(trainX, dtype= "float") / 255.0
    trainY = np.array(trainY)
    

    #Make training set
    print("Making test set:")
    base_dir = "Datasets"
    base_dir = os.path.join(base_dir, "cartoon_set_test")
    file_path = os.path.join(base_dir, "labels.csv")
    image_dir = os.path.join(base_dir, "img")
    haarcascade_eye_dir = "B2"
    haarcascade_eye_dir = os.path.join(haarcascade_eye_dir, "haarcascade_eye.xml")
    labels_file = open(file_path)
    lines = labels_file.readlines()
    detector = cv2.CascadeClassifier(haarcascade_eye_dir)
    testX = []
    testY = []
    failnum = 0
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir, line.split('\t')[3].replace('\n', ''))
        img = cv2.imread(image_path)
        rects = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(28, 28),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            failnum += 1
            continue
        for (fX, fY, fW, fH) in rects:
            roi = img[fY:fY + fH, fX:fX + fW]  
            roi = cv2.resize(roi, (50, 50))
            testX.append(roi)
            break
        testY.append(int(line.split('\t')[1]))
    testX = np.array(testX, dtype= "float") / 255.0 
    testY = np.array(testY)
    
    trainX = torch.from_numpy(np.transpose(trainX, (0,3,1,2)))
    testX = torch.from_numpy(np.transpose(testX, (0,3,1,2)))
    trainY = torch.from_numpy(trainY)
    testY = torch.from_numpy(testY)

    return trainX, testX, trainY, testY

# Define network structure
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
        )

        self.classifier = nn.Sequential(#input 64*52*42
            nn.Dropout(0.5),
            nn.Linear(18432, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5),
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
    model_path = "B2"
    model_path = os.path.join(model_path, 'eye_color_CNN_%03d.pth' % (epoch + 1))
    torch.save(model.state_dict(), model_path)
    if device == 'cuda':
        torch.cuda.empty_cache()


def test_CNN(TestX, TestY, path='./B2/eye_color_CNN.pth'):
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