''' Imports '''
import get_images
import get_landmarks
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

''' Load the data and their labels '''
image_directory = 'H:\\Fall 2020\\biometrics\\proj\\RonKauerTestFrames'
X, y = get_images.get_images(image_directory)

''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)

''' kNN classification treating every sample as a query'''
# initialize the classifie
knn_accuracy = []
NB_accuracy = []

print()
print("KNN")
for a in [1,3,5,7]:
    knn = KNeighborsClassifier(n_neighbors=a, metric = 'euclidean') 
    num_correct = 0
    labels_correct = []
    num_incorrect = 0
    labels_incorrect = []    
    
    
    for i in range(0, len(y)):
        query_img = X[i, :]
        query_label = y[i]
        
        template_imgs = np.delete(X, i, 0)
        template_labels = np.delete(y, i)
            
        # Set the appropriate labels
        # 1 is genuine, 0 is impostor
        y_hat = np.zeros(len(template_labels))
        y_hat[template_labels == query_label] = 1
        y_hat[template_labels != query_label] = 0
        
        knn.fit(template_imgs, y_hat) # Train the classifier
        y_pred = knn.predict(query_img.reshape(1,-1)) # Predict the label of the query
        
        # Print results
        if y_pred == 1:
            num_correct += 1
            labels_correct.append(query_label)
        else:
            num_incorrect += 1
            labels_incorrect.append(query_label)
    
    # Print results
    print()
    knn_accuracy.append(num_correct/(num_correct+num_incorrect))
    print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
          % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))    

print()
print("NB")
for a in [1,3,5,7]:
    gnb = GaussianNB() 
    num_correct_NB = 0
    labels_correct_NB = []
    num_incorrect_NB = 0
    labels_incorrect_NB = []
    
    for i in range(0, len(y)):
        query_img = X[i, :]
        query_label = y[i]
        
        template_imgs = np.delete(X, i, 0)
        template_labels = np.delete(y, i)
            
        # Set the appropriate labels
        # 1 is genuine, 0 is impostor
        y_hat = np.zeros(len(template_labels))
        y_hat[template_labels == query_label] = 1 
        y_hat[template_labels != query_label] = 0
        
        # knn.fit(template_imgs, y_hat) # Train the classifier
        # y_pred = knn.predict(query_img.reshape(1,-1)) # Predict the label of the query
        y_pred = gnb.fit(template_imgs, y_hat).predict(query_img.reshape(1,-1))
        
        # Print results
        if y_pred == 1:
            num_correct_NB += 1
            labels_correct_NB.append(query_label)
            # print(query_label)
            # plt.imshow(template_imgs)
        else:
            num_incorrect_NB += 1
            labels_incorrect_NB.append(query_label)
            # plt.imshow(X[i])
    
    # Print results
    # for i in labels_incorrect:
    #     print(i)
    
    print()
    NB_accuracy.append(num_correct/(num_correct+num_incorrect))
    print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
          % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))

print()
for i in range(0,len(NB_accuracy)):
    temp = (NB_accuracy[i]+knn_accuracy[i])/2
    print("Accuracy: %0.2f" %  temp)
    
    
    