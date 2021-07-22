#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:18:46 2020
@author: Jiani
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from scipy import interp
import matplotlib.pyplot as plt
from math import sqrt  
from sklearn.metrics import roc_curve, auc
import argparse
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from sklearn.model_selection import KFold
from numpy import linalg as LA
def Orig_data(XSD_path):
    XSD = pd.read_excel(XSD_path,header=None).values
    XSD_mask = np.ones_like(XSD)
    XSD_mask[np.isnan(XSD)] = 0
    XSD_mask_arr = XSD_mask.flatten()
    known_samples = np.nonzero(XSD_mask_arr)[0]   
    return XSD,XSD_mask,XSD_mask_arr,known_samples

def Combine_X(XSD_train_matrix,XSS,XSD_train_mask,XDD):
    XSS_mask = np.ones_like(XSS)
    XDD_mask = np.ones_like(XDD)
    X_row1 = np.hstack((XSS,XSD_train_matrix))
    X_row2 = np.hstack((XSD_train_matrix.T,XDD))
    X = np.vstack((X_row1,X_row2))
    X_mask_row1 = np.hstack((XSS_mask,XSD_train_mask))
    X_mask_row2 = np.hstack((XSD_train_mask.T,XDD_mask))
    X_mask = np.vstack((X_mask_row1,X_mask_row2))
    return X, X_mask

def compute_GSTtol(lam,p):
    pow1 = 1 / (2-p)
    pow2 = (p-1) / (2-p)
    root = 2 * lam *(1-p)
    gsttol = np.power(root,pow1) + lam*p *np.power(root,pow2)
    return gsttol

def GMST(A,lam,p):
    U,s,VT = LA.svd(A,full_matrices=False)
    gst_tol = compute_GSTtol(lam,p)
    gst_sigma_arr = np.zeros(len(s))
    for i in range(len(s)):
        sigma = s[i]        
        if sigma>gst_tol:
            gst_sigma = sigma 
            for k in range(2):
                gst_sigma = sigma - lam * p *np.power(gst_sigma,p-1)
        else: 
            gst_sigma = 0                 

        gst_sigma_arr[i] = gst_sigma  
    #print("gst_sigma",gst_sigma_arr)
    S = np.diag(gst_sigma_arr)
    gmst_tmp = np.dot(U,S)
    gmst = np.dot(gmst_tmp,VT)
    return gmst

def X_update(Y2,mu,M,A,X0,X0_mask):
    X1 = Y2 + mu*M
    invmat = (1+mu)*np.identity(A.shape[0])+np.dot(A,A.T)-A.T-A
    X2 = LA.inv(invmat)
    X_tmp = np.dot(X1,X2)
    X_tmp_arr = X_tmp.flatten() #value
    X0_arr = X0.flatten()    #value
    X0_mask_arr = X0_mask.flatten() #mask
    exist_id = np.nonzero(X0_mask_arr)
    X_tmp_arr[exist_id] = X0_arr[exist_id]
    X_tmp_arr[X_tmp_arr>1]=1
    X_tmp_arr[X_tmp_arr<0]=0
    X = X_tmp_arr.reshape((X0.shape[0],X0.shape[1]))
    return X

def M_update(mu,X,Y2,p):
    lam = 1/mu
    mat = X - lam*Y2
    M = GMST(mat,lam,p)
    return M
    
def C_update(A,Y1,lamda,mu):
    thresh = lamda/mu
    mat = A + (1/mu) * Y1 
    C1 = mat - thresh
    C1[C1<0] = 0
    C2 = mat + thresh
    C2[C2>0] = 0 
    C = C1 + C2
    return C

def A_update(X,mu,Y1,C):
    invmat = np.dot(X.T,X) + mu* np.identity(X.shape[0])
    A1 = LA.inv(invmat)
    A2 = np.dot(X.T,X) - Y1 + mu*C
    A = np.dot(A1,A2)    
    return A


def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-p', '--p_topofallfeature', type=float, nargs='?', default=0.8,help='Performing k-fold for cross validation.')
    parser.add_argument('-lamda', '--lamda_topofallfeature', type=float, nargs='?', default=0.25,help='Performing k-fold for cross validation.')
    parser.add_argument('-mu', '--mu_topofallfeature', type=float, nargs='?', default=4,help='Performing k-fold for cross validation.')
    parser.add_argument('-delta', '--delta_topofallfeature', type=float, nargs='?', default=0.01,help='Performing k-fold for cross validation.')
    parser.add_argument('-alpha', '--alpha_topofallfeature', type=float, nargs='?', default=0.4,help='Performing k-fold for cross validation.')
    parser.add_argument('-err_th', '--err_th_topofallfeature', type=float, nargs='?', default=1e-2,help='Performing k-fold for cross validation.')
    return parser.parse_args() 


if __name__=="__main__":
    args=parse_args()
    p =  args.p_topofallfeature
    lamda = args.lamda_topofallfeature
    mu = args.mu_topofallfeature
    delta = args.delta_topofallfeature    
    alpha = args.alpha_topofallfeature    
    err_th = args.err_th_topofallfeature    
    XSD_path = "./Site_Disease_mat.xlsx"
    XSD,XSD_mask,XSD_mask_arr,known_samples = Orig_data(XSD_path)
    XSS_jaccard = pd.read_excel("./Site_Jaccard_Sim_mat.xlsx",header=None).values
    XSS_cos = pd.read_excel("./Site_Cos_sim_mat.xlsx",header=None).values    
    XSS = alpha*XSS_jaccard + (1-alpha)*XSS_cos
    XDD = pd.read_excel("./Disease_Sim_mat_V2.xlsx",header=None).values
    kf = KFold(n_splits =10, shuffle=True)      #10 fold
    train_all=[]
    test_all=[]
    for train_ind,test_ind in kf.split(known_samples):  
        train_all.append(train_ind) 
        test_all.append(test_ind) 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold_int in range(10):
        print('fold_int',fold_int)
        XSD_train_id = train_all[fold_int]
        XSD_test_id = test_all[fold_int]
        XSD_train = known_samples[XSD_train_id]
        XSD_test = known_samples[XSD_test_id]
        XSD_train_list = np.zeros_like(XSD_mask_arr)
        XSD_train_list[XSD_train] = 1
        XSD_test_list = np.zeros_like(XSD_mask_arr)
        XSD_test_list[XSD_test] = 1         
        XSD_train_mask = XSD_train_list.reshape((XSD.shape[0],XSD.shape[1]))
        XSD_test_mask = XSD_test_list.reshape((XSD.shape[0],XSD.shape[1]))
        XSD_train_matrix = XSD_train_mask
        X0, X0_mask = Combine_X(XSD_train_matrix,XSS,XSD_train_mask,XDD)   
        lastX = lastM = lastA = lastC = lastY1 = lastY2 = np.zeros((X0.shape[0],X0.shape[0]))
        for i in range(500): 
            #print("iteration",i)
            currentX = X_update(lastY2,mu,lastM,lastA,X0,X0_mask)
            currentM = M_update(mu,currentX,lastY2,p)
            currentA = A_update(currentX,mu,lastY1,lastC)
            currentC = C_update(currentA,lastY1,lamda,mu)
            currentY1 = lastY1 + delta*(currentA-currentC)
            currentY2 = lastY2 + delta*(currentM-currentX)
            err_X = np.max(abs(currentX-lastX))
            err_M = np.max(abs(currentM-lastM))
            err_XM = np.max(abs(currentX-currentM))
            print("err_X",err_X)
            print("err_M",err_M)
            print("err_XM",err_XM)
            if (err_X < err_th) and (err_M < err_th) and (err_XM < err_th):
                break
            else:
                lastX = currentX
                lastM = currentM
                lastA = currentA
                lastC = currentC
                lastY1 = currentY1
                lastY2 = currentY2
                
        XSD_hat = currentX[0:741,741:918]  
        XSD_hat_arr=XSD_hat.flatten()                 #flattten prediction 
        candidate_index=np.where(XSD_mask_arr==0)[0]   
        candidate_samples=XSD_hat_arr[candidate_index]
        test_index=np.where(XSD_test_list==1)[0]
        test_samples=XSD_hat_arr[test_index]
        candidate_labels=np.zeros_like(candidate_samples)
        test_labels=np.ones_like(test_samples)
        scores=np.hstack((test_samples,candidate_samples))
        labels=np.hstack((test_labels,candidate_labels))
        fpr,tpr,threshold=roc_curve(labels,scores,pos_label=1)
        interp_y=interp(mean_fpr, fpr, tpr)
        tprs.append(interp_y)      
        tprs[-1][0]=0.0
        roc_auc = auc(fpr, tpr)   #roc_auc of each fold
        print('roc',roc_auc)
        aucs.append(roc_auc)
        plt.plot(fpr,tpr,lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (fold_int, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)   #auc of mean
    std_auc = np.std(aucs)        #5 auc      
    print('mean_auc',mean_auc)
    print('std_auc',std_auc) 
    #integration all ROCs 
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("spBLRSR.png")   #savefig_title

     
            
                        
                         
            
