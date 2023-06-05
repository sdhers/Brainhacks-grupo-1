import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
# matplotlib qt

#%%
# Import position data (obtained from DLC in a .h5 file)
ruta = 'C:/Users/dhers/Desktop/Brainhacks/Brainhacks-grupo-1/Motion_Tracking/'

file = ruta + 'DataDLC/videos_eval/TS - Caja 1 - A_L - Position.h5'

positions = pd.read_hdf(file)['DLC_resnet50_brainhackMay25shuffle1_230000']

positions.head()

#%%

# Plot the positions of the objects during the video (they should be constant)

plt.plot(positions['obj_1']['x'], positions['obj_1']['y'],".")
# plt.plot( [np.mean(positions['obj_1']['x'])], [np.mean(positions['obj_1']['y'])],"o")
plt.plot( [np.median(positions['obj_1']['x'])], [np.median(positions['obj_1']['y'])],"+")

plt.plot(positions['obj_2']['x'], positions['obj_2']['y'],".")
# plt.plot( [np.mean(positions['obj_2']['x'])], [np.mean(positions['obj_2']['y'])],"o")
plt.plot( [np.median(positions['obj_2']['x'])], [np.median(positions['obj_2']['y'])],"+")

#%%

# Import the label data (labeled using Label_videos.py)
labels = pd.read_csv(ruta + 'DataDLC/videos_eval/TS - Caja 1 - A_L - Labels.csv')
labels.head()


#%% 

# Simplify the data frame
df = pd.DataFrame()

for key in positions.keys():
    if key[1] != "likelihood":
        df[str( key[0] )+"_"+str( key[1] )] = positions[key]
    
df["label"] = labels["Left"] - labels["Right"]


df.to_csv(ruta + "prueba2.csv")

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

# xo1 = obj_1['x']
# yo1 = obj_1['y']

# xo2 = obj_2['x']
# yo2 = obj_2['y']

# Graph
plt.plot(xn[1000:1100], yn[1000:1100])
plt.plot(xh[1000:1100], yh[1000:1100])
# plt.plot(xo1[1000:1100], yo1[1000:1100])
# plt.plot(xo2[1000:1100], yo2[1000:1100])

#%%
# Write positions as points in R²
dots_n = np.dstack((xn, yn))[0]
dots_h = np.dstack((xh, yh))[0]

# Calculate distances nose - object
dist_1 = np.linalg.norm(dots_n - obj_1, axis=1)
dist_2 = np.linalg.norm(dots_n - obj_2, axis=1)

# Filter points close to object 1
max_distance = 80 # pixels (1 cm = 20 pixels)
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

ax.plot(towards_1[:, 0], towards_1[:, 1], ".", label = "Oriented towards object 1", color = "orange")
ax.plot(towards_2[:, 0], towards_2[:, 1], ".", label = "Oriented towards object 2", color = "purple")

ax.plot([x1], [y1], "o", lw = 20, label = "Object 1", color = "red")
ax.plot([x2], [y2], "o", lw = 20, label = "Object 2", color = "blue")

ax.axis('equal')

ax.legend(bbox_to_anchor = (0, 0.2, 1, 1), ncol=3, fancybox=True, shadow=True)
   
#%%

# Label frames of a video given a distance and angle threshold

L = len(dist_1)

max_degree = 50
max_distance = 70 # pixels

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

# clas_bosque = BA_model.predict( df2.drop(["label"], axis = 1))

a, b = 0,-1 # Set start and finish frames

plt.figure()
plt.plot( auto_label[a:b]*10, ".", color = "r", label = "Auto")
plt.plot( manual_label[a:b]*15, "o", color = "y", label = "Manual")
# plt.plot( clas_bosque[a:b]*60, color = "b", label = "Bosque")
plt.plot(angle_1[a:b], color = "g", label = "Orientation 1")
plt.plot(dist_1[a:b], color = "b", label = "Distance 1" )
plt.plot(angle_2[a:b]*(-1), color = "g", label = "Orientation 1")
plt.plot(dist_2[a:b]*(-1), color = "b", label = "Distance 1" )
plt.legend()
plt.show()


#%%
'''
#plt.plot(angulo / np.pi)
#plt.plot(v0[:, 0], v0[:, 1])
punto = 1200
plt.plot(v1[punto, 0], v1[punto, 1], ".")
plt.plot(v0[punto, 0], v0[punto, 1], ".")
plt.axhline(0)
plt.axvline(0)
#plt.plot(obj1[:, 0], obj1[:, 1], "o")


print(angulo[punto])
print(np.dot(v0[punto], v1[punto]))

print(v0.shape)
print(v1.shape)

print(np.rad2deg(np.arccos(np.dot([np.sqrt(2) / 2, np.sqrt(2) / 2], [1, 0]))))
print(v0[punto], v1[punto])
print(np.linalg.norm(v0).shape)
print(np.linalg.norm(v1, axis=1).shape)
print(np.repeat([[], []], 2, axis=1))

v0

v0 = obj1 - puntosC
v1 = puntosN - puntosC

print(v0.shape, v1.shape)
'''