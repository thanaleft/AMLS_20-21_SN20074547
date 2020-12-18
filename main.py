import os
import A1.gender_classification as A1
import A2.smile_detection as A2
import B1.face_shape_classification as B1
import B2.eye_color_classification as B2
from sklearn.metrics import classification_report,accuracy_score

#--------------------A1-------------------------
TrainX, TestX, TrainY, TestY = A1.make_dataset()
PredY = A1.Grid_SVM(TrainX, TrainY, TestX)
print("Task A1 classification report:\n%s\n"
      % (classification_report(TestY, PredY)))
A1_acc = accuracy_score(TestY, PredY)
del TrainX, TestX, TrainY, TestY

#-------------------A2--------------------------
# Haar cascade method:
# A2.CV_smile()

TrainX, TestX, TrainY, TestY = A2.make_dataset()
# A2.train_CNN(TrainX, TrainY)
model_path = "A2"
model_path = os.path.join(model_path, "Haar+CNN.pth")
A2_acc = A2.test_CNN(TestX, TestY, model_path)
del TrainX, TestX, TrainY, TestY

#------------------B1---------------------------
# face_recognition + SVM:
# TrainX, TestX, TrainY, TestY = B1.make_dataset_svm()
# PredY = A1.Grid_SVM(TrainX, TrainY, TestX)
# print("Task A1 classification report:\n%s\n"
#       % (classification_report(TestY, PredY)))

# # CNN method:
TrainX, TestX, TrainY, TestY = B1.make_dataset()
# B1.train_CNN(TrainX, TrainY)
model_path = "B1"
model_path = os.path.join(model_path, "face_shape_CNN.pth")
B1_acc = B1.test_CNN(TestX, TestY, model_path)
del TrainX, TestX, TrainY, TestY

#-----------------B2----------------------------
TrainX, TestX, TrainY, TestY = B2.make_dataset()
# B2.train_CNN(TrainX, TrainY)
model_path = "B2"
model_path = os.path.join(model_path, "eye_color_CNN.pth")
B2_acc = B2.test_CNN(TestX, TestY, model_path)
del TrainX, TestX, TrainY, TestY

print("Task\tTest_acc")
print("A1\t{:.2f}%".format(100*A1_acc))
print("A2\t{:.2f}%".format(A2_acc))
print("B1\t{:.2f}%".format(B1_acc))
print("B2\t{:.2f}%".format(B2_acc))