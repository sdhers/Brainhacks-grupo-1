import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

#%%
# Importar datos
ruta = 'C:/Users/dhers/Desktop/Brainhacks/Brainhacks-grupo-1/Motion_Tracking/DataDLC/videos_eval'

file = ruta + 'TS - Caja 1 - A_L - Position.h5'

df = pd.read_hdf('TS - Caja 1 - A_L - Position.h5')['DLC_resnet50_brainhackMay25shuffle1_230000']

df.head()

# plt.plot(df['obj_1']['x'],df['obj_1']['y'])
# plt.plot( [np.mean(df['obj_1']['x'])], [np.mean(df['obj_1']['y'])],"o")

plt.plot(df['obj_2']['x'],df['obj_2']['y'])
plt.plot( [np.mean(df['obj_2']['x'])], [np.mean(df['obj_2']['y'])],"o")

#%%

labels = pd.read_csv('TS - Caja 1 - A_L - Labels.csv')
labels.head()

#%%
# Extraer datos de la nariz
nariz = df['nose'].copy()

# Extraer datos de la cabeza
cabeza = df['neck'].copy()

# Extraer posiciones de los objetos
x1 = df['obj_1'].mean()['x']
y1 = df['obj_1'].mean()['y']
obj1 = np.repeat([[x1, y1]], len(nariz), axis=0)

x2 = df['obj_2'].mean()['x']
y2 = df['obj_2'].mean()['y']
obj2 = np.repeat([[x2, y2]], len(nariz), axis=0)


#%%
# Extaer posiciones en los dos ejes
xn = nariz['x']
yn = nariz['y']

xc = cabeza['x']
yc = cabeza['y']

# Graficar
plt.plot(xn[1000:1100], yn[1000:1100])
plt.plot(xc[1000:1100], yc[1000:1100])

#%%
# Escribir las posiciones como puntos en R²
puntosN = np.dstack((xn, yn))[0]
puntosC = np.dstack((xc, yc))[0]

# Calcular distancias de cada punto a cada objeto
dist1 = np.linalg.norm(puntosN - obj1, axis=1)
dist2 = np.linalg.norm(puntosN - obj2, axis=1)

# Filtrar puntos cercanos al objeto 1
distanciaMaxima = 80 # pixels (4 cm)
cercanos1 = puntosN[dist1 < distanciaMaxima]

# Filtrar puntos cercanos al objeto 2
cercanos2 = puntosN[dist2 < distanciaMaxima]
#%%
# Graficar objetos y puntos cercanos
fig, ax = plt.subplots()

ax.plot(xn, yn, ".", label = "Todos los puntos")

ax.plot(cercanos1[:, 0], cercanos1[:, 1], ".", label = "Cercanos a obj1", color = "orange")
ax.plot(cercanos2[:, 0], cercanos2[:, 1], ".", label = "Cercanos a obj2", color = "purple")

ax.plot([x1], [y1], "o", lw = 20, label = "Objeto 1", color = "red")
ax.plot([x2], [y2], "o", lw = 20, label = "Objeto 2", color = "blue")

ax.add_artist(Circle((x1, y1), distanciaMaxima, color = "grey", alpha = 0.3))
ax.add_artist(Circle((x2, y2), distanciaMaxima, color = "grey", alpha = 0.3))

ax.axis('equal')

ax.legend(bbox_to_anchor = (0, 0.2, 1, 1), ncol=3, fancybox=True, shadow=True)

#%%
'''
El criterio completo debería ser:

1. Nariz cercana al objeto.

2. Que la nariz esté más cerca que la cola (o que la cabeza).

3. Parametrizar la recta nariz-cola (o nariz-cabeza) y fijarnos si interseca al objeto (nos dice que lo está mirando).
'''

#%%
v0_ = puntosN - puntosC
v1_ = obj1 - puntosC
v2_ = obj2 - puntosC


#v0 = v0 / np.repeat(np.linalg.norm(v0, axis=1), 2, axis=1)
v0 = np.dstack(( v0_[:,0] / np.linalg.norm(v0_, axis=1) ,  v0_[:,1] / np.linalg.norm(v0_, axis=1) ))[0]  
v1 = np.dstack(( v1_[:,0] / np.linalg.norm(v1_, axis=1) ,  v1_[:,1] / np.linalg.norm(v1_, axis=1) ))[0]
v2 = np.dstack(( v2_[:,0] / np.linalg.norm(v2_, axis=1) ,  v2_[:,1] / np.linalg.norm(v2_, axis=1) ))[0]  

angulo1 = np.zeros(len(v0))
angulo2 = np.zeros(len(v0))


for i in range(len(v0)):
    angulo1[i] = np.arccos( np.dot( v1[i], v0[i] ) )*180/np.pi
    angulo2[i] = np.arccos( np.dot( v2[i], v0[i] ) )*180/np.pi
    

L = len(dist1)
umbralA = 50
umbralD = 80 # pixels

clasificador = np.zeros( L )
for frame in range(L):
    if angulo1[frame] < umbralA and dist1[frame] < umbralD:
        clasificador[frame] = 1
    elif angulo2[frame] < umbralA and dist2[frame] < umbralD:
        clasificador[frame] = 2     
    else:
         clasificador[frame] = 0  


#%%

clasificador_labels = np.zeros( L )
for frame in range(L):
    if labels["Left"][frame] == 1:
        clasificador_labels[frame] = 1
    elif labels["Right"][frame] == 1:
        clasificador_labels[frame] = 2     
    else:
         clasificador[frame] = 0

#%%

a, b = 0,-1
plt.figure()
plt.plot( clasificador[a:b]*50, color = "r", label = "Auto")
plt.plot( clasificador_labels[a:b]*55, color = "g", label = "Manual")
plt.plot(angulo1[a:b], label = "Angulo1")
plt.plot(angulo2[a:b] + 1000, label = "Angulo2")
plt.plot( dist1[a:b], label = "Distancia1" )
plt.plot( dist2[a:b] + 1000, label = "Distancia2" )
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