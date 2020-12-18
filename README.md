# README
##1 Required Package
* scikit-learn
* PyTorch
* tqdm
* dlib
* face_recognition
* matplotlib
* cv2(opencv-python)

##2 Structure and tips
* In main.py, I run each task model in turn. In some tasks, I implemented more than one model, but I only present the selected in main.py. If you want to test other model, you can uncomment those line.
* in each task, I first prepare and preprocesse the dataset. In my project I will consider the data from **Datasets/celeca** and **Datasets/cartoon_set** as training set and separate 0.2 from training set as validation set. The images in **Datasets/celeca_test** and **Datasets/cartoon_set__test** will be considered to be test set.
* Task A1 is using traditional machine learning algorithm. So, in the main function it present training and test as well. You have to fill the **Datasets/celeca** with required images for training.
* Other tasks are using CNN, which I save my pre-trained model in each task folder and the main function will not present the training step(I comment all code related to training), the programme will load the model and directly run the test step.
* If you want to train the model, just uncomment the training code in main.py. The trained model will be saved in the task folder with different name of my model.
* You can change the model path in main.py to select the model you want.
* When testing B1, I strongly recommand you to use 16G RAM laptop, since 8G laptop may test the model in very low speed. (At least my 8G Mac takes much more longer than my 16G Windows laptop(only used few seconds), both laptop are used CPU to test)
*  If you want to train a new model, please use GPU, that will save a lot of time.