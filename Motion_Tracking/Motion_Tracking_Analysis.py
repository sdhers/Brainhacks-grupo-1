'''
Automatic labeling
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import os

#%%

# Specify the folder path
ruta = 'C:/Users/dhers/Desktop/Brainhacks/Brainhacks-grupo-1/Motion_Tracking/DataDLC/videos_eval/'

# Get the list of file names in the folder
file_names = os.listdir(ruta)

lista_h5 = []
lista_labels = []

# Print the file names
for file_name in file_names:
    if "Position.h5" in file_name:
        lista_h5.append( file_name )
    elif "Labels.csv" in file_name:
        lista_labels.append( file_name )
        
#%%

# Separate one video to evaluate the labeling
video = 18
p_test = lista_h5.pop(video - 1)
l_test = lista_labels.pop(video - 1)

#%%

# Import position data (obtained from DLC in a .h5 file)
positions = pd.read_hdf(ruta + p_test)['DLC_resnet50_brainhackMay25shuffle1_230000']
positions.head()

# Import the label data (labeled using Label_videos.py)
labels = pd.read_csv(ruta + l_test)
labels.head()

#%%

# Plot the positions of the objects during the video (they should be constant)

plt.plot(positions['obj_1']['x'], positions['obj_1']['y'],".")
# plt.plot( [np.mean(positions['obj_1']['x'])], [np.mean(positions['obj_1']['y'])],"o")
plt.plot( [np.median(positions['obj_1']['x'])], [np.median(positions['obj_1']['y'])],"+")

plt.plot(positions['obj_2']['x'], positions['obj_2']['y'],".")
# plt.plot( [np.mean(positions['obj_2']['x'])], [np.mean(positions['obj_2']['y'])],"o")
plt.plot( [np.median(positions['obj_2']['x'])], [np.median(positions['obj_2']['y'])],"+")


#%% 

# Simplify the data frame
df = pd.DataFrame()

for key in positions.keys():
    if key[1] != "likelihood":
        df[str( key[0] )+"_"+str( key[1] )] = positions[key]
    
df["label"] = labels["Left"] - labels["Right"]


df.to_csv(ruta + 'Data_frames/' + "df -" + l_test)

#%%
# Extract nose positions
nose = positions['nose'].copy()

# Extract head positions
head = positions['head'].copy()

# Extract object positions
x1 = positions['obj_1'].median()['x']
y1 = positions['obj_1'].median()['y']
obj_1 = np.repeat([[x1, y1]], len(nose), axis = 0)

x2 = positions['obj_2'].median()['x']
y2 = positions['obj_2'].median()['y']
obj_2 = np.repeat([[x2, y2]], len(nose), axis = 0)


#%%

# Extract positions on both axis
xn = nose['x']
yn = nose['y']

xh = head['x']
yh = head['y']

# Graph
plt.plot(xn[1000:1100], yn[1000:1100])
plt.plot(xh[1000:1100], yh[1000:1100])

#%%

# Write positions as points in R²
dots_n = np.dstack((xn, yn))[0]
dots_h = np.dstack((xh, yh))[0]

# Calculate distances nose - object
dist_1 = np.linalg.norm(dots_n - obj_1, axis=1)
dist_2 = np.linalg.norm(dots_n - obj_2, axis=1)

# Filter points close to object 1
max_distance = 60 # pixels (1 cm = 20 pixels)
close_to_1 = dots_n[dist_1 < max_distance]

# Filter points close to object 2
close_to_2 = dots_n[dist_2 < max_distance]

#%%

# Graph objects and close points
fig, ax = plt.subplots()

ax.plot(xn, yn, ".", label = "Every dot in video")

ax.plot(close_to_1[:, 0], close_to_1[:, 1], ".", label = "Close to object 1", color = "orange")
ax.plot(close_to_2[:, 0], close_to_2[:, 1], ".", label = "Close to object 2", color = "purple")

ax.plot([x1], [y1], "o", lw = 20, label = "Object 1", color = "red")
ax.plot([x2], [y2], "o", lw = 20, label = "Object 2", color = "blue")

ax.add_artist(Circle((x1, y1), max_distance, color = "grey", alpha = 0.3))
ax.add_artist(Circle((x2, y2), max_distance, color = "grey", alpha = 0.3))

ax.axis('equal')

ax.legend(bbox_to_anchor = (0, 0.2, 1, 1), ncol=3, fancybox=True, shadow=True)

#%%

# Lets calculate the angle comprehended between the line head-nose and the line head-object
v0_ = dots_n - dots_h
v1_ = obj_1 - dots_h
v2_ = obj_2 - dots_h


#v0 = v0 / np.repeat(np.linalg.norm(v0, axis=1), 2, axis=1)
v0 = np.dstack(( v0_[:,0] / np.linalg.norm(v0_, axis=1) ,  v0_[:,1] / np.linalg.norm(v0_, axis=1) ))[0]  
v1 = np.dstack(( v1_[:,0] / np.linalg.norm(v1_, axis=1) ,  v1_[:,1] / np.linalg.norm(v1_, axis=1) ))[0]
v2 = np.dstack(( v2_[:,0] / np.linalg.norm(v2_, axis=1) ,  v2_[:,1] / np.linalg.norm(v2_, axis=1) ))[0]  

angle_1 = np.zeros(len(v0))
angle_2 = np.zeros(len(v0))


for i in range(len(v0)):
    angle_1[i] = np.arccos( np.dot( v1[i], v0[i] ) )*180/np.pi
    angle_2[i] = np.arccos( np.dot( v2[i], v0[i] ) )*180/np.pi

# Filter points of orientation towards object 1
max_degree = 50 # degrees
towards_1 = dots_n[angle_1 < max_degree]

# Filter points of orientation towards object 2
towards_2 = dots_n[angle_2 < max_degree]

#%%

# Graph objects and low angle points
fig, ax = plt.subplots()

ax.plot(xn, yn, ".", label = "Every dot in video")

ax.plot(towards_1[:, 0], towards_1[:, 1], ".", label = "Oriented towards object 1", color = "orange", alpha = 0.25)
ax.plot(towards_2[:, 0], towards_2[:, 1], ".", label = "Oriented towards object 2", color = "purple", alpha = 0.25)

ax.plot([x1], [y1], "o", lw = 20, label = "Object 1", color = "red")
ax.plot([x2], [y2], "o", lw = 20, label = "Object 2", color = "blue")

ax.add_artist(Circle((x1, y1), max_distance, color = "grey", alpha = 0.8))
ax.add_artist(Circle((x2, y2), max_distance, color = "grey", alpha = 0.8))

ax.axis('equal')

ax.legend(bbox_to_anchor = (0, 0.2, 1, 1), ncol=3, fancybox=True, shadow=True)
   
#%%

# Label frames of a video given a distance and angle threshold

L = len(dist_1)

max_degree = 45
max_distance = 60 # pixels 

auto_label = np.zeros( L )

for frame in range(L):
    if angle_1[frame] < max_degree and dist_1[frame] < max_distance:
        auto_label[frame] = 1
    elif angle_2[frame] < max_degree and dist_2[frame] < max_distance:
        auto_label[frame] = -1     
    else:
        auto_label[frame] = 0  


#%%

# Read the manual labeling for the chosen video

manual_label = np.zeros( L )

for frame in range(L):
    if labels["Left"][frame] == 1:
        manual_label[frame] = 1
    elif labels["Right"][frame] == 1:
        manual_label[frame] = -1     
    else:
         manual_label[frame] = 0
         
#%%
'''
# Graph manual vs auto labeling
a, b = 0, -1 # Set start and finish frames

plt.figure()
plt.plot( auto_label[a:b]*10, ".", color = "r", label = "Auto")
plt.plot( manual_label[a:b]*15, "o", color = "y", label = "Manual")
plt.plot(angle_1[a:b], color = "g", label = "Orientation 1")
plt.plot(dist_1[a:b], color = "b", label = "Distance 1" )
plt.plot(angle_2[a:b]*(-1), color = "g", label = "Orientation 1")
plt.plot(dist_2[a:b]*(-1), color = "b", label = "Distance 1" )
plt.legend()
plt.show()
'''

#%%
'''
Random forest
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

#%%

# Import data to train the random forest
data = pd.DataFrame(columns = ['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y', 'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y', 'R_ear_x', 'R_ear_y', 'head_x', 'head_y', 'neck_x', 'neck_y', 'body_x', 'body_y', 'tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y', 'label'])

for i in range( len( lista_h5 ) ):
    file = ruta + lista_h5[i]
    fileL = ruta + lista_labels[i]

    df = pd.read_hdf(file)['DLC_resnet50_brainhackMay25shuffle1_230000']
    labels = pd.read_csv(fileL)
    
    data0 = pd.DataFrame()
    
    for key in df.keys():
        if key[1] != "likelihood":
            if key[0] == "obj_1" or key[0] == "obj_2":
                data0[str( key[0] )+"_"+str( key[1] )] = [df[key].median()]*len(df[key])
            else:
                data0[str( key[0] )+"_"+str( key[1] )] = df[key]
        
    data0["label"] = labels["Left"] - labels["Right"]

    data = pd.concat([data, data0], ignore_index = True)

print(data.shape)
# A data frame containing the positions and labels for all videos but one (that will be used for testing)

#%% 

# lets now prepare the testing data
file = ruta + p_test

df = pd.read_hdf(file)['DLC_resnet50_brainhackMay25shuffle1_230000']
labels = pd.read_csv(ruta + l_test)

test = pd.DataFrame()

for key in df.keys():
    if key[1] != "likelihood":
        test[str( key[0] )+"_"+str( key[1] )] = df[key]
    
test["label"] = labels["Left"] - labels["Right"]

print(test.shape)
# A data frame containing the positions and labels for the testing video

#%%

# Dividimos los datos en entrenamiento y prueba

# X son nuestras variables independientes
X = data.drop(["label"], axis = 1)

# y es nuestra variable dependiente
y = data.label.astype('int')

# División 75% de datos para entrenamiento, 25% de daatos para test
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#%%

# Create the Random Forest model (and set the number of estimators (decision trees))
BA_model = RandomForestClassifier(n_estimators = 60, random_state = 42, max_depth = 30, class_weight = "balanced")

BA_model.fit(X, y)

#%%

BA_label = BA_model.predict( test.drop(["label"], axis = 1))

#%%
# BA_label_2 = [BA_label[0]]

# for i in range(1,7499):
#     valor = BA_label[i]
#     if BA_label[i-1] == 1 and BA_label[i+1] == 1:
#         valor = 1
#     elif BA_label[i-1] == -1 and BA_label[i+1] == -1:
#         valor = -1
#     BA_label_2.append(valor)
    
# BA_label_2.append(BA_label[-1])   

# print(len(BA_label_2))

#%%
# BA_label_prom = np.convolve(BA_label,[1/5]*5, "same")

# BA_label_3 = np.zeros(7500)
# BA_label_3[BA_label_prom < - 0.3] = -1
# BA_label_3[0.3 < BA_label_prom] = 1

#%%

a, b = 0, -1 # Set start and finish frames

plt.figure()
plt.plot( manual_label[a:b]*6, "o", color = "k", label = "Manual")
# plt.plot( np.array(BA_label_3[a:b])*6, "o", color = "c", label = "Bosque 2")
plt.plot( BA_label[a:b]*4, "o", color = "y", label = "RF Model")
plt.plot( auto_label[a:b]*2, "o", color = "m", label = "Auto")
plt.plot(angle_1[a:b], color = "g", label = "Orientation 1")
plt.plot(dist_1[a:b]/2.5, color = "b", label = "Distance 1" )
plt.plot(angle_2[a:b]*(-1), color = "g", label = "Orientation 2")
plt.plot(dist_2[a:b]/2.5*(-1), color = "b", label = "Distance 2" )
plt.legend()
plt.show()

#%%

plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 18
# Compute areas and colors

a, b = 200, -1 # Set start and finish frames

colors1 = ['red' if label == 1 else 'gray' for label in manual_label[a:b]]
colorsBA1 = ['blue' if label == 1 else 'gray' for label in BA_label[a:b]]

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter((angle_1[a:b]+90)/180*np.pi, dist_1[a:b]/20, c = colors1, s = 12, alpha=0.6)
c = ax.scatter(-(angle_1[a:b]-90)/180*np.pi, dist_1[a:b]/20, c = colorsBA1, s = 12, alpha=0.6)

ang_plot = np.linspace(np.pi/4,np.pi/2,25).tolist()

c = ax.plot([0]+ang_plot+[0],[0]+[3]*25+[0],c="k", linestyle='dashed', linewidth = 4)


# plt.yscale("log")
plt.ylim([0,4])
plt.yticks([1,2,3,4],["1 cm","2 cm","3 cm","4 cm"])
plt.xticks([0, 45/180*np.pi,90/180*np.pi,135/180*np.pi,np.pi,225/180*np.pi,270/180*np.pi,315/180*np.pi],["  90°","45°", "0°","45°","90°  ", "135°    ","180°","    135°"]   )
plt.show()

#%%

plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 18
# Compute areas and colors

a, b = 200, -1 # Set start and finish frames

colors2 = ['red' if label == -1 else 'gray' for label in manual_label[a:b]]
colorsBA2 = ['blue' if label == -1 else 'gray' for label in BA_label[a:b]]

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter((angle_2[a:b]+90)/180*np.pi, dist_2[a:b]/20, c = colors2, s = 12, alpha=0.6)
c = ax.scatter(-(angle_2[a:b]-90)/180*np.pi, dist_2[a:b]/20, c = colorsBA2, s = 12, alpha=0.6)

ang_plot = np.linspace(np.pi/4,np.pi/2,25).tolist()

c = ax.plot([0]+ang_plot+[0],[0]+[3]*25+[0],c="k", linestyle='dashed', linewidth = 4)


# plt.yscale("log")
plt.ylim([0,4])
plt.yticks([1,2,3,4],["1 cm","2 cm","3 cm","4 cm"])
plt.xticks([0, 45/180*np.pi,90/180*np.pi,135/180*np.pi,np.pi,225/180*np.pi,270/180*np.pi,315/180*np.pi],["  90°","45°", "0°","45°","90°  ", "135°    ","180°","    135°"]   )
plt.show()

#%%

suma_auto1_error = 0
suma_auto1_falso = 0
suma_auto1 = 0
suma_bosque1_error = 0
suma_bosque1_falso = 0
suma_bosque1 = 0
eventos = 0


for i in range(len(manual_label)):
    if manual_label[i] != 0 and auto_label[i] == 0:
        suma_auto1_error += 1
    if manual_label[i] == 0 and auto_label[i] != 0:
        suma_auto1_falso += 1
    if manual_label[i] != 0 and auto_label[i] != 0:
        suma_auto1 += 1
    if manual_label[i] != 0 and BA_label[i] == 0:
        suma_bosque1_error += 1        
    if manual_label[i] == 0 and BA_label[i] != 0:
        suma_bosque1_falso += 1
    if manual_label[i] != 0 and BA_label[i] != 0:
        suma_bosque1 += 1
    if manual_label[i] != 0:
        eventos += 1
    

print(eventos/len(manual_label))

print(suma_auto1/len(manual_label))
print(suma_bosque1/len(manual_label))

print(suma_auto1/eventos)
print(suma_bosque1/eventos)

print(eventos/len(manual_label))
print(suma_auto1_falso/eventos)
print(suma_bosque1_falso/eventos)
