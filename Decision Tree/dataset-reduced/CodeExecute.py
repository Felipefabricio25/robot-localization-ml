from skimage import io
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder



folder = '/home/admsistemas/Documents/ReducedDatasetGMaps/*/*.png'

folder_test = '/home/admsistemas/Documents/ReducedDatasetGMaps/test_map.png'
folder_test2 = '/home/admsistemas/Documents/ReducedDatasetGMaps/test_map2.png'
folder_test3 = '/home/admsistemas/Documents/ReducedDatasetGMaps/test_map3.png'

reserve = []

for i in range(36963):
    reserve.append(205)

cv_img = []
actual_img = []
fldr = []
parallel = []
x_append = []
y_label = []
i = 0
i_qual = 0

#---

n= cv2.imread(folder_test)
n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT_create()
# kp, dc = sift.detectAndCompute(n,None)
# for things in dc:
#     for stuff in things:
#         parallel.append(stuff)

# actual_img.append(list(parallel))
# parallel = []

# dst = cv2.cornerHarris(n,2,3,0.04)
# for things in dst:
#     for stuff in things:
#         parallel.append(stuff)

# actual_img.append(list(parallel))
# parallel = []

orb = cv2.ORB_create()
keypoints = orb.detect(n, None)
keypoints, descriptors = orb.compute(n, keypoints)
x_append.append(descriptors)
y_label.append(folder_test)

n= cv2.imread(folder_test2)
n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT_create()
# kp, dc = sift.detectAndCompute(n,None)
# for things in dc:
#     for stuff in things:
#         parallel.append(stuff)

# actual_img.append(list(parallel))
# parallel = []

# dst = cv2.cornerHarris(n,2,3,0.04)
# for things in dst:
#     for stuff in things:
#         parallel.append(stuff)
# actual_img.append(list(parallel))
# parallel = []

orb = cv2.ORB_create()
keypoints = orb.detect(n, None)
keypoints, descriptors = orb.compute(n, keypoints)
x_append.append(descriptors)
y_label.append(folder_test2)

n= cv2.imread(folder_test3)
n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT_create()
# kp, dc = sift.detectAndCompute(n,None)
# for things in dc:
#     for stuff in things:
#         parallel.append(stuff)

# actual_img.append(list(parallel))
# parallel = []

# dst = cv2.cornerHarris(n,2,3,0.04)
# for things in dst:
#     for stuff in things:
#         parallel.append(stuff)
# actual_img.append(list(parallel))
# parallel = []

orb = cv2.ORB_create()
keypoints = orb.detect(n, None)
keypoints, descriptors = orb.compute(n, keypoints)
x_append.append(descriptors)
y_label.append(folder_test3)

x_image_features = np.vstack(np.array(x_append))

scaler = MinMaxScaler(feature_range=(0,1))
x_image_scaled = scaler.fit_transform(x_image_features)

kmeans = KMeans(n_clusters=6, random_state=0).fit(x_image_scaled)

bovw_v_entrance = np.zeros([len(x_append), 6])
for index, features in enumerate(x_append):
  for i in kmeans.predict(features):
    bovw_v_entrance[index, i] +=1
    continue
  continue
  break

print(bovw_v_entrance)



#---

    #print(len(n.reshape(-1).tolist()))

for img in glob.glob(folder):
    '''
    im=Image.open(img)

    pixels = list(im.getdata())
    cv_img.append(pixels)
    '''
    
    n= cv2.imread(img)
    n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)

    # sift = cv2.SIFT_create()
    # kp, dc = sift.detectAndCompute(n,None)
    # if dc is not None:
    #     for things in dc:
    #         for stuff in things:
    #             parallel.append(stuff)

    # n = n.tolist()

    # print(dc)

    # dst = cv2.cornerHarris(n,2,3,0.04)
    # for things in dst:
    #     for stuff in things:
    #         parallel.append(stuff)
    # n = dst.tolist()

    orb = cv2.ORB_create()
    keypoints = orb.detect(n, None)
    keypoints, descriptors = orb.compute(n, keypoints)
    parallel = []
    n = descriptors

    # if i_qual != 0:
    #     print(f"new {img}")
    #     i_qual += 1
    #     if i_qual == 7:
    #         i_qual = 0
    # if n.count(n[0]) == len(n): 
    #     #i_qual = 1
    #     print(img)
    # else:
    if descriptors is None:
        pass
    else:
        i+=1
        print(i)  
        cv_img.append(descriptors)
        #print(n)
        fldr.append(img)

    #print(len(n.reshape(-1).tolist()))
    #parallel = []

x_image_features = np.vstack(np.array(cv_img))
scaler = MinMaxScaler(feature_range=(0,1))
x_image_scaled = scaler.fit_transform(x_image_features)

kmeans = KMeans(n_clusters=6, random_state=0).fit(x_image_scaled)

bovw_v = np.zeros([len(cv_img), 6])
for index, features in enumerate(cv_img):
  for i in kmeans.predict(features):
    bovw_v[index, i] +=1
    continue
  continue
  break

le = LabelEncoder()
y_label_le = le.fit_transform(fldr)

percent = []
scale = []
scale2 = []

for i in range(342):
    percent.append(0)
    scale.append(i)

for i in range(36963):
    scale2.append(i)

print("... pixels?")

print(len(cv_img))

# dfX = pd.DataFrame(cv_img)
# dfX.fillna(0)
# dfA = pd.DataFrame(actual_img)
# dfA.fillna(0)

# print(dfX)

# print(dfA)

#print(fldr)

final = []

for things in fldr:
    smth = things.replace('/home/admsistemas/Documents/ReducedDatasetGMaps/','')
    smth = smth.replace('.png','')
    smth = smth.replace('botanic_map_','')
    smth = smth.replace('/',',')

    smtha = smth.split(",")

    smth = smtha[1]

    #print(smth)

    if '_' in smth:
        smth = smth.split('_')
        value = float(smth[0])
        noise = float(smth[2])
        #print(f"Value = {value}, Noise = {noise}")
    else:
        value = float(smth)
        noise = 0
        #print(f"Value = {value}, Noise = {noise}")

    if value - int(value) != 0:
        value1 = value+0.5
        value2 = value-0.5

        if value1 == 342.0:
            value1 = 341.0

        #print(value,value1,value2)

        # copyper = percent
        # copyper[int(value1)] = int(50) - int(noise*5)
        # copyper[int(value2)] = int(50) - int(noise*5)

        final.append(int(value1))
    else:
        # copyper = percent
        # copyper[int(value)] = int(100) - int(noise*10)

        final.append(int(value))

    #print(final)

    i = 0
    # for things in copyper:
    #     copyper[i] = 0
    #     i+=1

print("saiu!!")

#print(cv_img)

dfY = pd.DataFrame(final)
dfY.fillna(0)
print(dfY)
# print(dfA)
# dfA.fillna(0)

print(bovw_v)

X_train, X_test, y_train, y_test = \
    train_test_split(bovw_v, dfY, test_size=0.2, 
                     random_state=0)

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=50, 
                                random_state=1,
                                n_jobs=2)

forest.fit(X_train, y_train)

'''

pipe_forest = make_pipeline(StandardScaler(),
                            RandomForestClassifier(random_state=1))

param_range = [2,3,4,5,6,7,8]
param_est = [10,20,30,40,50,60,70,80,90,100]

param_grids = [{'randomforestclassifier__max_depth': param_range, 
                'randomforestclassifier__criterion':["entropy"],
                'randomforestclassifier__n_estimators':param_est},
                {'randomforestclassifier__max_depth': param_range,
                'randomforestclassifier__criterion':["gini"],
                'randomforestclassifier__n_estimators':param_est}]

gs = GridSearchCV(estimator=pipe_forest, 
                  param_grid=param_grids, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

'''


y_result = forest.predict(X_test)

print(y_result)

for things in y_result:
    # i = 0
    # for stuff in things:
    #     if stuff != 0:
    #         print(f"Percent in the {i} Quadrant: {stuff}%")
    #     i+=1
    print(f"Located in the {things} quadrant!")
    print("-----")

