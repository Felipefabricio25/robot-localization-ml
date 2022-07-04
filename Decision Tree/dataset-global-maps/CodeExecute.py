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
from PIL import Image

folder = '/home/admsistemas/Documents/DatasetGlobalMaps/*/*.png'

cv_img = []
fldr = []
i = 0
for img in glob.glob(folder):
    '''
    im=Image.open(img)

    pixels = list(im.getdata())
    cv_img.append(pixels)

    i+=1
    print(i)

    '''
    
    n= cv2.imread(img)
    cv_img.append(n.reshape(-1).tolist())
    #print(n)
    fldr.append(img)
    
percent = []
scale = []
scale2 = []

for i in range(342):
    percent.append(0)
    scale.append(i)

for i in range(36963):
    scale2.append(i)

print("... pixels?")

dfX = pd.DataFrame(cv_img, columns=scale2)

print(dfX)

#print(fldr)

final = []

for things in fldr:
    smth = things.replace('/home/admsistemas/Documents/DatasetGlobalMaps/','')
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
        copyper[int(value1)] = int(50) - int(noise*2)
        copyper[int(value2)] = int(50) - int(noise*2)

        final.append(list(copyper))
    else:
        copyper = percent
        copyper[int(value)] = int(100) - int(noise*4)

        final.append(list(copyper))

    #print(final)

    i = 0
    for things in copyper:
        copyper[i] = 0
        i+=1

print("saiu!!")

#print(cv_img)

dfY = pd.DataFrame(final, columns=scale)
print(dfY)

X_train, X_test, y_train, y_test = \
    train_test_split(dfX, dfY, test_size=0.2, 
                     random_state=0)

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=50, 
                                random_state=1,
                                n_jobs=2)

forest.fit(X_train, y_train)

y_result = forest.predict(X_test)

print(y_result)

for things in y_result:
    i = 0
    for stuff in things:
        if stuff != 0:
            print(f"Percent in the {i} Quadrant: {stuff}%")
        i+=1
    print("-----")

# plot_decision_regions(X_train, y_train, 
#                       classifier=forest, test_idx=range(105, 150))

# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.title("Random Forest")
# plt.tight_layout()
# #plt.savefig('images/03_22.png', dpi=300)
# plt.show()

#img = io.imread('file_path')
