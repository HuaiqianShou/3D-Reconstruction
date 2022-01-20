"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import copy
import matplotlib.pyplot as plt
import helper
import util
from scipy import signal
import scipy.optimize
import random

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''




def eightpoint(pts1, pts2, M):

    N  = len(pts1)
    a = np.zeros(9)
    data1 = copy.deepcopy(pts1)
    data2 = copy.deepcopy(pts2)
    normd_pts1 = data1/M
    normd_pts2 = data2/M
    A = []
    A = np.array(A)
    for i in range(N):
        x1, y1 = normd_pts1[i, 0], normd_pts1[i, 1]
        x2, y2 = normd_pts2[i, 0], normd_pts2[i, 1]
        a[0] = x2*x1
        a[1] = x2*y1
        a[2] = x2
        a[3] = y2*x1
        a[4] = y2*y1
        a[5] = y2
        a[6] = x1
        a[7] = y1
        a[8] = 1
    
        A = np.concatenate((A,a),axis = 0)
    A = A.reshape(N,9)

    #print ("A",A.shape)
    U1,S1,V1 = np.linalg.svd(A)
    eignvector = V1.T[:,-1]
    eig = np.array(eignvector)
    F = eig.reshape(3,3)   
    F = util.refineF(F, normd_pts1, normd_pts2)
    F = util._singularize(F)
    Trans = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = (Trans.T)@F@Trans

    return F
    




'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    
    
    return (K2.T)@F@K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    
    # refer to some information from http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    
    N = pts1.shape[0]
    A = np.zeros((4,4))
    P = np.zeros((N,3))
    c1 = C1
    c2 = C2
    jb  = pts1[0][1]
    print('hahahaha')
    print(jb*(c1[2,:]))
    for i in range(N):
        x1 = pts1[i][0]
        x2 = pts2[i][0]
        y1 = pts1[i][1]
        y2 = pts2[i][1]
#        A[0,:] = np.array([y1*c1[2,:]-c1[1,:]])
#        A[1,:] = np.array([c1[0,:]-x1*c1[2,:]])
#        A[2,:] = np.array([y2*c2[2,:]-c2[1,:]])
#        A[3,:] = np.array([c2[0,:]-x2*c2[2,:]])
        A[0,:] = np.array([- c1[0,:] + x1*c1[2,:]])
        A[1,:] = np.array([y1*c1[2,:] - c1[1,:]])
        A[2,:] = np.array([-c2[0,:]+x2*c2[2,:]])
        A[3,:] = np.array([y2*c2[2,:]-c2[1,:]])
        U1,S1,V1=np.linalg.svd(A)
        eignvector = V1.T[:,3]
        eig = np.array(eignvector)
        #print(eig.shape)
        P[i,:] = (eig[0:3]/eig[3])
        #print(eig.shape)
    
    P_err = np.vstack((P.T,np.ones((1,N))))
    err = 0
    for i in range(N):
        
        bar1 = C1@(P_err[:,i])
        bar2 = C2@(P_err[:,i])
        bar1 = bar1/bar1[-1]
        bar2 = bar2/bar2[-1]
        pts1_new = np.vstack((pts1.T,np.ones((1,N))))
        pts2_new = np.vstack((pts2.T,np.ones((1,N))))
        err = err + np.sum((bar1-pts1_new[:,i])**2+((bar2-pts2_new[:,i])**2))

    return P, err 
        
    


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x1, y1 = int(np.round(x1)), int(np.round(y1))
    point = [x1,y1,1]
    point = (np.array(point)).T
    line = F@point
    scale = np.linalg.norm(line)
    line = line/scale
    n = 5
    window1 = im1[y1-n:y1+n+1,x1-n:x1+n+1]
    a = line[0]
    b = line[1]
    c = line[2]
    
    h,w,d = im2.shape
    
    y = np.arange(h)
    
    sim_min = 10000
    gkern1d = signal.gaussian(11, std=5).reshape(11, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    weight = []
    weight = np.array(weight)
    weight = np.repeat(gkern2d[:, :, np.newaxis], 3, axis=2)
    
    
    for i in range(h):
        y = i
        x = int((-b*y-c)/a)
        if(x >= n and x + n < w and y >= n and y + n < h and x1 >= n and x1 + n < w and y1 >= n and y1 + n < h):
            
            window2 = im2[y-n:y+n+1,x-n:x+n+1]
            sim_cur = np.sum((window1-window2)**2*weight)
            if  sim_cur < sim_min:
                sim_min = sim_cur
                x2 = x
                y2 = y

    return x2, y2
        
        


'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=500, tol=4):

    
    max_inliers = -1
    F_output, inliers = None, None
    N = pts1.shape[0]
    for i in range(nIters):
        
        random_num = random.sample(range(0, N), 8)


        set1 = pts1[random_num,:]
        set2 = pts1[random_num,:]
        
        F = eightpoint(set1, set2, M)
        diff_set = []
        for i in range(N):
            x1 = pts1[i,0]
            y1 = pts1[i,1]
            x2 = pts2[i,0]
            y2 = pts2[i,1]
            point = [x1,y1,1]
            point2 = [x2,y2,1]
            point2 = np.array(point2)
            point = (np.array(point)).T
            line = F@point

            scale2 = np.sqrt(np.sum(line[:2] ** 2, axis=0))
 #           print(scale,scale2)
            line = line/scale2
            
            a = line[0]
            b = line[1]
            c = line[2]
            diff = abs(a*x2+b*y2+c)

            diff_set.append(diff)
            
            
        
        difference = np.array(diff_set)

        a = (difference < tol)
        
        count  = np.sum(a)
        if count > max_inliers:
            max_inliers = count
            F_output = F
            inliers = a

        print(max_inliers)
        
    F_output = eightpoint(pts1[inliers,:], pts2[inliers,:], M)
        
    return F_output, inliers

    


            

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    
    
    if theta ==0:
        
        return np.identity(3)
        
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)    
    K = r/theta
    Kx = K[0]
    Ky = K[1]
    Kz = K[2]
    
    Km = np.zeros((3,3))
    Km[0,1] = -Kz
    Km[0,2] = Ky
    Km[1,0] = Kz
    Km[1,2] = -Kx
    Km[2,0] = -Ky
    Km[2,1] = Kx
    
    Iden = np.identity(3)
    sq = np.linalg.matrix_power(Km,2)
    R = Iden + sin_theta* Km + (1-cos_theta)*sq
    
    return R
    
    

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    
    
    
    

    theta=np.arccos(1/2*(np.trace(R)-1))
    sin_theta =np.sin(theta)
    Rm = np.zeros(3)
    Rm[0] = R[2][1]-R[1][2]
    Rm[1] = R[0][2]-R[2][0]
    Rm[2] = R[1][0]-R[0][1]
    w =(1/(2*sin_theta))*Rm

    r=w*theta
    return r
    

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    
    t2 = x[-3:]
    r2 = x[-6:-3]
    P = x[:-6]
    P = np.array(P)
    P = P.reshape(P.shape[0]//3,3)
    t2 = np.array(t2)
    t2 = t2.reshape(3,1)
    
    R2 = rodrigues(r2)
    
    C1 = K1 @ M1
    M2 = np.concatenate((R2,t2),axis = 1)
    C2 = K2 @ M2
    N = P.shape[0]
    p1_homo = np.concatenate((P,np.ones((N,1))),axis = 1)
    p1_homo = p1_homo.T
    p1_hat = C1@p1_homo
    p1_hat = p1_hat[0:2,:]/p1_hat[2,:]
    p1_hat = p1_hat.T
    p2_homo = np.concatenate((P,np.ones((N,1))),axis = 1)
    p2_homo = p2_homo.T
    p2_hat = C2@p2_homo
    p2_hat = p2_hat[0:2,:]/p2_hat[2,:]
    p2_hat = p2_hat.T
    
    residuals = np.concatenate([(p1-p1_hat).reshape([-1]),(p2-p2_hat).reshape([-1])])
    return residuals
   


'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    

    R2 = M2_init[:,0:3]
    
    
    t2 = M2_init[:,3]
    
    
    r2 = invRodrigues(R2)
    
    r2 = np.array(r2)
    t2 = np.array(t2)
    
    
    #print(t2.shape,r2.shape,(P_init.flatten()).shape)
    
    
    x = np.concatenate((P_init.reshape([-1]),r2,t2))
    
    
    
    
    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()
    
    
    
#    output = scipy.optimize.leastsq(func, x)
    
    output = scipy.optimize.minimize(func, x).x
    
    t2_output = output[-3:] 
    r2_output = output[-6:-3] 
    P = output[:-6]

    

    N = P.shape[0] // 3
    
    P2 = P.reshape(N,3)

    r2 = r2_output.reshape(3,1)
    
    t2 = t2_output.reshape(3,1)
    
    R2 = rodrigues(r2)
    
    M2 = np.hstack((R2, t2))

    return M2, P2
    
    
    