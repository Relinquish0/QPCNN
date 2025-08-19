import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

psi_positive = np.array([[1], [0]])
psi_negative = np.array([[0], [1]])
Jx = np.array([[0, 1], [1, 0]])
Jy = np.array([[0, -1j], [1j, 0]])
Jz = np.array([[1, 0], [0, -1]])
I = np.eye(2)

def cal(p1, p2):
    """
    Calculate the probability of a projection result
    :param p1: Projection matrix of the first particle
    :param p2: Projection matrix of the second particle
    :return: indicates the probability
    """
    entangle = np.kron(psi_negative, psi_negative) - 1*np.kron(psi_positive, psi_positive)  # An entangled matrix of two particles. Varying this for a weaker entangled state.
    prob = np.dot(entangle.T, np.dot(np.kron(p1, p2), entangle)) / 2
    probability = np.real(prob[0][0])
    return probability

def uncorrelated(n,excel_path):
    data = {
        'xa': [], 'ya': [], 'za': [],
        'xb': [], 'yb': [], 'zb': [],
        'p00': [], 'p01': [],
        'p10': [], 'p11': []
    }
    for x in range(n):
        # 随机生成方向a
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        xa = np.sin(theta) * np.cos(phi)
        ya = np.sin(theta) * np.sin(phi)
        za = np.cos(theta)

        Sa1 = (I + (xa * Jx + ya * Jy + za * Jz)) / 2  # Projective matrix when a = 1 
        Sa0 = (I - (xa * Jx + ya * Jy + za * Jz)) / 2  # Projective matrix when a = 0 

        p1_ = cal(Sa1, I)  # Probability when a = 1
        p0_ = cal(Sa0, I)  # Probability when a = 0
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        xb = np.sin(theta) * np.cos(phi)
        yb = np.sin(theta) * np.sin(phi)
        zb = np.cos(theta)

        Sb1 = (I + (xb * Jx + yb * Jy + zb * Jz)) / 2  # Projective matrix when b = 1 
        Sb0 = (I - (xb * Jx + yb * Jy + zb * Jz)) / 2  # Projective matrix when b = 0 

        p_1 = cal(I, Sb1)  # Probability when b = 1
        p_0 = cal(I, Sb0)  # Probability when b = 0
        data['xa'].append(xa)
        data['ya'].append(ya)
        data['za'].append(za)
        data['xb'].append(xb)
        data['yb'].append(yb)
        data['zb'].append(zb)
        data['p00'].append(p0_*p_0)
        data['p01'].append(p0_*p_1)
        data['p10'].append(p1_*p_0)
        data['p11'].append(p1_*p_1)
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

def correlated(n,mode,excel_path): # mode = 2 or 5
    data = {
        'xa': [], 'ya': [], 'za': [],
        'xb': [], 'yb': [], 'zb': [],
        'p00': [], 'p01': [],
        'p10': [], 'p11': []
    }

    for num in range(n):
        # Random generated directiona
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        xa = np.sin(theta) * np.cos(phi)
        ya = np.sin(theta) * np.sin(phi)
        za = np.cos(theta)

        # Random generated directionb
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        xb = np.sin(theta) * np.cos(phi)
        yb = np.sin(theta) * np.sin(phi)
        zb = np.cos(theta)

        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        xc = np.sin(theta) * np.cos(phi)
        yc = np.sin(theta) * np.sin(phi)
        zc = np.cos(theta)

        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        xd = np.sin(theta) * np.cos(phi)
        yd = np.sin(theta) * np.sin(phi)
        zd = np.cos(theta)
        for i in range(1,mode,1):
            if i == 1:#A0B0
                x1 = xa
                y1 = ya
                z1 = za
                x2 = xb
                y2 = yb
                z2 = zb
            if i == 2:#A0B1
                x1 = xa
                y1 = ya
                z1 = za
                x2 = xd
                y2 = yd
                z2 = zd
            if i == 3:#A1B0
                x1 = xc
                y1 = yc
                z1 = zc
                x2 = xb
                y2 = yb
                z2 = zb     
            if i == 4:#A1B1
                x1 = xc
                y1 = yc
                z1 = zc
                x2 = xd
                y2 = yd
                z2 = zd   
            Sa1 = (I + (x1 * Jx + y1 * Jy + z1 * Jz)) / 2  # Measure the projection matrix of particle 1 as 1 in direction a
            Sa0 = (I - (x1 * Jx + y1 * Jy + z1 * Jz)) / 2  # Measure the projection matrix of particle 1 as 0 in direction a
            Sb1 = (I + (x2 * Jx + y2 * Jy + z2 * Jz)) / 2  # Measure the projection matrix of particle 2 as 1 in direction b
            Sb0 = (I - (x2 * Jx + y2 * Jy + z2 * Jz)) / 2  # Measure the projection matrix of particle 2 as 0 in direction b
            p1_ = cal(Sa1, I)  # Just measure the probability that particle 1 will result in 1
            p11 = cal(Sa1, Sb1)  # Measure the probability that particles 1 and 2 will both result in 1
            p10 = cal(Sa1, Sb0)  # The probability that measuring particles 1 and 2 results in 1, 0, respectively
            p0_ = cal(Sa0, I)  # Only measure the probability that particle 1 will result in 0
            p01 = cal(Sa0, Sb1)  # Measure the probability that particles 1 and 2 result in 0,1, respectively
            p00 = cal(Sa0, Sb0)  # Measure the probability that particles 1 and 2 will both result in 0

            data['xa'].append(x1)
            data['ya'].append(y1)
            data['za'].append(z1)
            data['xb'].append(x2)
            data['yb'].append(y2)
            data['zb'].append(z2)
            data['p00'].append(p00)
            data['p01'].append(p01)
            data['p10'].append(p10)
            data['p11'].append(p11)
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

def count(N, input_file, output_file, if_count):
    f = pd.read_excel(input_file)
    f = f.to_numpy()#recieve theta in unfaked data

    coordinate = f[:,:6] 
    a_0 = np.zeros(coordinate.shape[0])
    a_1 = np.ones_like(a_0)
    coordinate = pd.DataFrame(coordinate)
    a_0 = pd.DataFrame(a_0)
    a_1 = pd.DataFrame(a_1)

    pro_00 = f[:,6]
    pro_01 = f[:,7]
    pro_10 = f[:,8]
    pro_11 = f[:,9]

    pro_measured_00 = np.zeros(coordinate.shape[0])
    pro_measured_01 = np.zeros(coordinate.shape[0])
    pro_measured_10 = np.zeros(coordinate.shape[0])
    pro_measured_11 = np.zeros(coordinate.shape[0])

    times_measured_00 = np.zeros(coordinate.shape[0])
    times_measured_01 = np.zeros(coordinate.shape[0])
    times_measured_10 = np.zeros(coordinate.shape[0])
    times_measured_11 = np.zeros(coordinate.shape[0])

    for i in range(0,coordinate.shape[0],1):
        random_values = np.random.choice([0, 1, 2, 3], size=N, p=[pro_00[i], pro_01[i],pro_10[i],pro_11[i]])
        times_measured_00[i] = np.sum(random_values==0)
        times_measured_01[i] = np.sum(random_values==1)
        times_measured_10[i] = np.sum(random_values==2)
        times_measured_11[i] = np.sum(random_values==3)

        pro_measured_00[i] = times_measured_00[i]/(times_measured_00[i]+times_measured_01[i])
        pro_measured_01[i] = 1-pro_measured_00[i]
        pro_measured_10[i] = times_measured_10[i]/(times_measured_10[i]+times_measured_11[i])
        pro_measured_11[i] = 1-pro_measured_10[i]
        # if i%10000 == 0:
        #     print(i,'finished')

    df00 = pd.DataFrame(pro_measured_00, columns=['Column1'])
    df01 = pd.DataFrame(pro_measured_01, columns=['Column2'])
    df10 = pd.DataFrame(pro_measured_10, columns=['Column1'])
    df11 = pd.DataFrame(pro_measured_11, columns=['Column2'])

    merged_df_a0 = pd.concat([coordinate, a_0, df00, df01], axis=1)
    merged_df_a1 = pd.concat([coordinate, a_1, df10, df11], axis=1)

    merged = pd.concat([merged_df_a0,merged_df_a1],axis=0)
    merged.to_excel(output_file, index=False)

    if if_count == True:
        gf00 = pd.DataFrame(times_measured_00, columns=['time00'])
        gf01 = pd.DataFrame(times_measured_01, columns=['time01'])
        gf10 = pd.DataFrame(times_measured_10, columns=['time10'])
        gf11 = pd.DataFrame(times_measured_11, columns=['time11'])

        merge = pd.concat([coordinate,gf00,gf01,gf10,gf11],axis=1)
        merge.to_excel('{}_count.xlsx'.format(output_file.removesuffix('.xlsx')), index=False)

def shuffle_excel(input_file):
    f = pd.read_excel(input_file)
    data = f.to_numpy()
    data = np.random.permutation(data)
    data = pd.DataFrame(data)
    data.to_excel(input_file,index=False)