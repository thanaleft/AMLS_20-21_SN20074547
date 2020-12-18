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

def make_dataset():
    base_dir = "./Datasets"
    #Make training set
    print("Making A1 training set.")
    file_path_train = os.path.join(base_dir, "celeba/labels.csv")
    image_dir_train = os.path.join(base_dir, "celeba/img")
    labels_file = open(file_path_train)
    lines = labels_file.readlines()
    trainX = []
    trainY = []
    fail_num = 0
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir_train, line.split('\t')[1])
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

    #Make test set
    print("Making A1 test set.")
    file_path_test = os.path.join(base_dir, "celeba_test/labels.csv")
    image_dir_test = os.path.join(base_dir, "celeba_test/img")
    labels_file = open(file_path_test)
    lines = labels_file.readlines()
    testX = []
    testY = []
    fail_num = 0
    i = 0
    for line in tqdm(lines[1:]):
        image_path = os.path.join(image_dir_test, line.split('\t')[1])
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

def Grid_SVM(x_train, y_train, x_test):
    print("Traing model:")
    param_grid = {'C': [1e1, 1e2],
              'gamma': [10, 100], 
              'decision_function_shape':["ovo"],
                "degree":[3,4]}
    model = GridSearchCV(svm.SVC(kernel='rbf', class_weight="balanced", verbose=True),
            param_grid, cv=5)
    model.fit(x_train, y_train)
    print(model.best_params_, model.best_score_)
    y_pred = model.best_estimator_.predict(x_test)
    return y_pred