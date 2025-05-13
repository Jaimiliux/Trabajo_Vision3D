#!/usr/bin/env python3
import numpy as np
import cv2
import os, sys, glob
import math
from scipy.linalg import svd
import scipy.optimize

# 1) Detect homographies and image points
def get_H(save_dir, images, world_pts):
    H_list, img_pts = [], []
    os.makedirs(save_dir, exist_ok=True)
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (7,7), None)
        print(f"{idx}: Checkerboard {'found' if found else 'NOT found'}")
        if not found: continue
        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 30,1e-6)
        )
        pts2d = corners.reshape(-1,2)
        H, _ = cv2.findHomography(world_pts, pts2d, cv2.RANSAC, 5.0)
        H_list.append(H); img_pts.append(pts2d)
        cv2.drawChessboardCorners(img, (7,7), corners, found)
        thumb = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
        cv2.imwrite(os.path.join(save_dir, f"{idx}_corners.png"), thumb)
    return H_list, img_pts

# 2) Zhang's v helper
def v(i,j,H):
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ], dtype=float)

# 3) Compute intrinsic K via SVD
def get_K(H_list):
    V = []
    for H in H_list:
        V.append(v(0,1,H))
        V.append(v(0,0,H)-v(1,1,H))
    V = np.array(V)
    if V.shape[0] < 2: raise RuntimeError("Need ≥2 homographies")
    _,_,Vt = np.linalg.svd(V)
    b = Vt[-1]
    # ensure B33 positive
    if b[5] < 0: b = -b
    B11,B12,B22,B13,B23,B33 = b
    # form B
    # compute v0,u0 per Zhang
    denom = (B11*B22 - B12**2)
    v0 = (B12*B13 - B11*B23)/denom
    u0 = (B12*B23 - B22*B13)/denom
    lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11
    alpha = math.sqrt(lam / B11)
    beta  = math.sqrt((lam*B11)/denom)
    gamma = 0.0  # assume zero skew
    K = np.array([[alpha, gamma, u0],
                  [0.0,   beta,  v0],
                  [0.0,   0.0,   1.0]])
    print("Linear intrinsic K:\n", K)
    return K

# 4) Extrinsics
def get_extrin(K, H_list):
    invK = np.linalg.inv(K)
    R_list, t_list = [], []
    for H in H_list:
        h1,h2,h3 = H[:,0], H[:,1], H[:,2]
        lam = 1.0/np.linalg.norm(invK.dot(h1))
        r1 = lam*invK.dot(h1); r2 = lam*invK.dot(h2); t = lam*invK.dot(h3)
        r3 = np.cross(r1,r2)
        R = np.vstack((r1,r2,r3)).T
        R_list.append(R); t_list.append(t)
    return R_list, t_list

# 5) Reprojection residuals
def residuals(x, img_pts, world_pts, R_list, t_list):
    alpha,beta,u0,v0,k1,k2 = x
    res = []
    K = np.array([[alpha,0.0,u0],[0.0,beta,v0],[0.0,0.0,1.0]])
    for pts2d,R,t in zip(img_pts,R_list,t_list):
        for (X,Y),(u_obs,v_obs) in zip(world_pts,pts2d):
            M = np.array([X,Y,1.0])
            p = R.dot(M)+t
            xh,yh = p[0]/p[2], p[1]/p[2]
            r2 = xh*xh + yh*yh
            xd,yd = xh*(1+k1*r2+k2*r2*r2), yh*(1+k1*r2+k2*r2*r2)
            uv = K.dot([xd,yd,1.0])
            u_pred,v_pred = uv[0]/uv[2], uv[1]/uv[2]
            res.append(u_pred-u_obs)
            res.append(v_pred-v_obs)
    return np.array(res)

# 6) Refine
def refine(K0, img_pts, world_pts, R_list, t_list):
    alpha,beta = K0[0,0], K0[1,1]
    u0,v0 = K0[0,2], K0[1,2]
    x0 = [alpha,beta,u0,v0, 0.0,0.0]
    sol = scipy.optimize.least_squares(
        residuals,x0,
        args=(img_pts,world_pts,R_list,t_list),
        method='lm',verbose=2)
    α,β,u0,v0,k1,k2 = sol.x
    K_opt = np.array([[α,0.0,u0],[0.0,β,v0],[0.0,0.0,1.0]])
    print("Refined K:\n",K_opt)
    print("Distortion k1,k2=",k1,k2)
    return K_opt,(k1,k2)

# 7) Mean error
def mean_error(K,dist,img_pts,world_pts,R_list,t_list):
    res = residuals([K[0,0],K[1,1],K[0,2],K[1,2],dist[0],dist[1]],
                    img_pts,world_pts,R_list,t_list)
    return np.mean(np.abs(res))

# MAIN
if __name__=='__main__':
    imgs = [cv2.imread(f) for f in glob.glob('./Muestra/*.jpeg')]
    xs,ys = np.meshgrid(range(7),range(7))
    world_pts = np.vstack((xs.ravel(),ys.ravel())).T.astype(float)*24.0
    H_list, img_pts = get_H('./Resultado',imgs,world_pts)
    if len(H_list)<2: sys.exit("Need ≥2 homographies")
    K0 = get_K(H_list)
    R_list,t_list = get_extrin(K0,H_list)
    K_opt,dist = refine(K0,img_pts,world_pts,R_list,t_list)
    err = mean_error(K_opt,dist,img_pts,world_pts,R_list,t_list)
    print(f"Mean reprojection error: {err:.4f} px")
