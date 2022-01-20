'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''




import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
intri = np.load('../data/intrinsics.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

pts1 = data['pts1']
pts2 = data['pts2']
k1 = intri['K1']
k2 = intri['K2']
print(k1.shape)


h,w,d = im1.shape
M = np.max((h,w,d))
F = sub.eightpoint(pts1, pts2, M)
E = sub.essentialMatrix(F, k1, k2)

M2s = helper.camera2(E)
#print(M2s.shape)
#for i in range(4):
#    print(M2s[:,:,i])
iden = np.identity(3)

M1 = np.concatenate((iden,np.zeros((3,1))),axis=1)
C1 = k1@M1
#print(C1.shape)
count = 0
for i in range(4):
    C2 =k2@M2s[:,:,i]
    
    P,_ = sub.triangulate(C1, pts1, C2, pts2)
    
    if(np.min(P[:,2])>0):
        count +=1
        M2 = M2s[:,:,i]
        C2_output = C2
        P_output = P
        
#np.savez('q3_3.npz', M2=M2, C2=C2_output, P=P_output)