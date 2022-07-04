import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Test set')

def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y  

img1 = cv2.imread('DatasetSimple/botanic_map.png')  
img2 = cv2.imread('DatasetSimple/test_map.png') 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

figure, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].imshow(img1, cmap='gray')
ax[1].imshow(img2, cmap='gray')


plt.imshow(img2),plt.show()
  

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

len(keypoints_1), len(keypoints_2)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

points = []
pointsX = []
pointsY = []
y_final = []

for match in matches:
  p1 = keypoints_1[match.queryIdx].pt
  p2 = keypoints_2[match.trainIdx].pt

  points.append(p1)

for X, Y in points:
    pointsX.append(X)
    pointsY.append(Y)

center = sum(pointsX)/len(pointsX), sum(pointsY)/len(pointsY)
plt.plot(center[0], center[1], marker='o')
plt.imshow(img1),plt.show()

pointsX = np.asarray(pointsX)
pointsY = np.asarray(pointsY)

h, w   = img1.shape
pcent = 0.01
h_scale_top = center[0] + h*pcent
h_scale_bot = center[0] - h*pcent
w_scale_top = center[1] + w*pcent
w_scale_bot = center[1] - w*pcent

for X,Y in points:
  if X >= h_scale_bot and X <= h_scale_top and Y >= w_scale_bot and Y <= w_scale_top:
    y_final.append(1)
  else:
    y_final.append(0)

dfX = pd.DataFrame(points)

print(h_scale_bot, h_scale_top, w_scale_bot, w_scale_top)
print(center, h*pcent, w*pcent)
print(dfX)
print(y_final)

X_train, X_test, y_train, y_test = \
    train_test_split(dfX, y_final, test_size=0.1, 
                     stratify=y_final,
                     random_state=0)

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=1,
                                max_depth=1,
                                random_state=1,
                                n_jobs=2)

# pipe_forest = make_pipeline(StandardScaler(),
#                             RandomForestClassifier(random_state=1))

# param_range = [1,2]
# param_est = [1,2,3,4,5,6,7,8,9,10]

# param_grids = [{'randomforestclassifier__max_depth': param_range, 
#                 'randomforestclassifier__criterion':["entropy"],
#                 'randomforestclassifier__n_estimators':param_est},
#                 {'randomforestclassifier__max_depth': param_range,
#                 'randomforestclassifier__criterion':["gini"],
#                 'randomforestclassifier__n_estimators':param_est}]

# gs = GridSearchCV(estimator=pipe_forest, 
#                   param_grid=param_grids, 
#                   scoring='accuracy', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)

forest.fit(X_train, y_train)

y_final = forest.predict(points)

print(y_final)

final_p = []

i = 0
for things in y_final:
  if things == 1:
    final_p.append(points[i])
    plt.plot(points[i][0], points[i][1], marker='o', c='blue')
  else:
    plt.plot(points[i][0], points[i][1], marker='o', c='red')
  i+=1

pointsX = []
pointsY = []

for X, Y in final_p:
    pointsX.append(X)
    pointsY.append(Y)

center = sum(pointsX)/len(pointsX), sum(pointsY)/len(pointsY)
plt.plot(center[0], center[1], marker='o')
plt.imshow(img1),plt.show()

pointsX = np.asarray(pointsX)
pointsY = np.asarray(pointsY)

# plot_decision_regions(X_train, y_train, 
#                       classifier=forest, test_idx=range(105, 150))

# plt.imshow(img1)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.title("Random Forest")
# plt.tight_layout()
# #plt.savefig('images/03_22.png', dpi=300)
# plt.show()



#plt.plot(pointsX, pointsY, 'o')

# parameters, covariance = curve_fit(Gauss, pointsX, pointsY)

# fit_A = parameters[0]
# fit_B = parameters[1]
  
# fit_y = Gauss(pointsX, fit_A, fit_B)

# print(fit_y)

#plt.plot(pointsX, pointsY, 'o', label='data')
# plt.plot(pointsX, fit_y, '-', label='fit')
# plt.legend()
# plt.show()


#print(points)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()