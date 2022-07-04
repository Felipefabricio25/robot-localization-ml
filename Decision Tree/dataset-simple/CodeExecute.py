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
from sklearn.preprocessing import label_binarize
from skimage import filters



folder = '/home/admsistemas/Documents/DatasetSimple/*/*.png'

folder_test = '/home/admsistemas/Documents/DatasetSimple/test_map.png'
folder_test2 = '/home/admsistemas/Documents/DatasetSimple/test_map2.png'
folder_test3 = '/home/admsistemas/Documents/DatasetSimple/test_map3.png'

reserve = []

for i in range(36963):
    reserve.append(205)

cv_img = []
actual_img = []
fldr = []
i = 0
i_qual = 0

quadrant_percent = []
quadrants = []

#---

n= cv2.imread(folder_test)
n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(n,2,3,0.04)
print(len(dst))
actual_img.append(dst.tolist())

n= cv2.imread(folder_test2)
n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(n,2,3,0.04)
print(len(dst))
actual_img.append(dst.tolist())

n= cv2.imread(folder_test3)
n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(n,2,3,0.04)
print(len(dst))
actual_img.append(dst.tolist())

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
    dst = cv2.cornerHarris(n,2,3,0.04)
    n = dst.tolist()


    # forbidden_len = [43200, 37632, 37296, 73926, 74181, 37218, 37473, 37728]

    # print(len(n))

    # n = n[:36963]
    # print(f"Resized to {len(n)}")

    # if i_qual != 0:
    #     print(f"new {img}")
    #     i_qual += 1
    #     if i_qual == 4:
    #         i_qual = 0
    if n.count(n[0]) == len(n):
        print(img)
    else:
        i+=1
        print(f"img number {i}, size {len(dst)}")  
        cv_img.append(dst.tolist())
        #print(n)
        fldr.append(img)

    #print(len(n.reshape(-1).tolist()))
    
percent = []
scale = []
scale2 = []
scale3 = []

for i in range(342):
    percent.append(0)
    scale.append(i)

for i in range(36963):
    scale2.append(i)

for i in range(255):
    scale2.append(i)

print("... pixels?")

for things in actual_img:
    print(len(things))

dfA = pd.DataFrame(actual_img)

print(dfA)

dfX = pd.DataFrame(cv_img)

print(dfX)

#print(fldr)

final = []

for things in fldr:
    smth = things.replace('/home/admsistemas/Documents/DatasetSimple/','')
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

        copyper = percent
        copyper[int(value1)] = int(50) - int(noise*5)
        copyper[int(value2)] = int(50) - int(noise*5)

        quadrant_percent.append(int(value))
        quadrant_percent.append(int(int(100) - int(noise*10)))

        final.append(list(copyper))
        quadrants.append(list(quadrant_percent))
    else:
        copyper = percent
        copyper[int(value)] = int(100) - int(noise*10)

        quadrant_percent.append(int(value))
        quadrant_percent.append(int(int(100) - int(noise*10)))

        final.append(list(copyper))
        quadrants.append(list(quadrant_percent))

    #print(final)

    i = 0
    for things in copyper:
        copyper[i] = 0
        i+=1
    i = 0
    quadrant_percent = []

print("saiu!!")

#print(cv_img)

dfY = pd.DataFrame(final)
dfYQ = pd.DataFrame(quadrants)
print(dfY)
print(dfYQ)
print(dfA)

X_train, X_test, y_train, y_test = \
    train_test_split(dfX, dfY, test_size=0.2, 
                     random_state=0)

X_trainQ, X_testQ, y_trainQ, y_testQ = \
    train_test_split(dfX, dfYQ, test_size=0.2, 
                     random_state=0)

# forest = RandomForestClassifier(criterion='entropy',
#                                 n_estimators=50, 
#                                 random_state=1,
#                                 n_jobs=2)

# forest.fit(X_train, y_train)

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

y_trainQ = label_binarize(y_trainQ, classes=[0, 1])

gs = gs.fit(X_trainQ, y_trainQ)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# y_result = forest.predict(dfA)

# print(y_result)

# for things in y_result:
#     i = 0
#     for stuff in things:
#         if stuff != 0:
#             print(f"Percent in the {i} Quadrant: {stuff}%")
#         i+=1
#     print("-----")

