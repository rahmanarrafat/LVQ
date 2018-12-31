# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:53:52 2018

@author: SHOLIHIN RAHMAN
"""

import numpy as np

'''
##################################################
data = data file yang akan dilakukan pelatihan
MaxEpoch = maksimal iterasi yang digunakan pada proses pelatihan
w = bobot awal yang digunakan pada proses pelatihan
label = kelas dari data yang akan dilakukan pelatihan
alpha = learining rate yang akan digunakan pada proses pelatihan
r_alpha = pengurangan learning rate
##################################################
'''

def LVQ(data, MaxEpoch, w, label, alpha, r_alpha):
    e = 0
    while(e < MaxEpoch):
        for i in range(data.shape[1]): #jumlah data
            jarak = np.zeros((1,w.shape[1]),dtype = float) 
            for j in range(w.shape[1]): #banyak kelas bobot
                for k in range(w.shape[0]): #banyak data/fitur
                    jarak[0, j] = jarak[0, j] + ( data[k,i] - w[k,j] )**2
            jarak = jarak**0.5            
            C = np.argmin(jarak)     
            jarak2 = w[:,C]
            if C == label[0,i]:
                jarak2 = jarak2 + (alpha * (data[:, i] - jarak2))
                w[:, C] = jarak2
            else:
                jarak2 = jarak2 - (alpha * (data[:, i] - jarak2))
                w[:, C] = jarak2
        alpha = alpha - (r_alpha * alpha)
        e = e+1    
        if e == MaxEpoch:
            break
    return w
