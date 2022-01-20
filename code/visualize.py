'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import submission as sub
import helper



data = np.load('../data/some_corresp.npz')
intri = np.load('../data/intrinsics.npz')
temp = np.load('../data/templeCoords.npz')
#print((tem['x1']).shape)
x1 = temp['x1']
y1 = temp['y1']
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

pts1 = data['pts1']
pts2 = data['pts2']
k1 = intri['K1']
k2 = intri['K2']
#print(k1.shape)


h,w,d = im1.shape
M = np.max((h,w,d))


F = sub.eightpoint(pts1, pts2, M)

points_out1 = np.zeros((x1.shape[0],2))
points_out2 = np.zeros((x1.shape[0],2)) 
points_out1[:,0] = x1[:,0]
points_out1[:,1] = y1[:,0]  

     
for i in range(x1.shape[0]):
    
    x2,y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i,:], y1[i,:])
    
    points_out2[i,0] = x2
    points_out2[i,1] = y2

E = sub.essentialMatrix(F, k1, k2)

M2s = helper.camera2(E)
iden = np.identity(3)

M1 = np.concatenate((iden,np.zeros((3,1))),axis=1)

c1 = k1@M1
#print(c1.shape)
count = 0
for i in range(4):
    c2 =k2@M2s[:,:,i]
    
    P,_ = sub.triangulate(c1, points_out1, c2, points_out2)
    
    if(np.min(P[:,2])>0):
        count +=1
        M2 = M2s[:,:,i]
        C2_output = c2
        P_output = P
        
np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=c1, C2=C2_output)
fig = plt.figure()


ax = fig.add_subplot(111, projection='3d')

ax.scatter(P_output[:, 0], P_output[:, 1], P_output[:, 2], c='b', marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()