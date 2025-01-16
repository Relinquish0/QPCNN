import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
f = pd.read_excel("C:/Users/86136/Desktop/dataset_generation/uncorrelated/data_bell_uncorrelated_train.xlsx")#train/validation
f = f.to_numpy()#recieve theta in unfaked data

coordinate = f[:,:6] 
a_0 = np.zeros(coordinate.shape[0])
a_1 = np.ones_like(a_0)
coordinate = pd.DataFrame(coordinate)
a_0 = pd.DataFrame(a_0)
a_1 = pd.DataFrame(a_1)

pro_a0 = f[:,6]
pro_a1 = f[:,7]
pro_b0 = f[:,8]
pro_b1 = f[:,9]

pro_measured_00 = np.zeros(pro_a0.shape[0])
pro_measured_01 = np.zeros(pro_a0.shape[0])
pro_measured_10 = np.zeros(pro_a0.shape[0])
pro_measured_11 = np.zeros(pro_a0.shape[0])

N = 100000#measurement rounds
for i in range(0,pro_a0.shape[0],1):
    random_values = np.random.choice([0, 1, 2, 3], size=N, p=[pro_a0[i]*pro_b0[i], pro_a0[i]*(1-pro_b0[i]),(1-pro_a0[i])*pro_b0[i],(1-pro_a0[i])*(1-pro_b0[i])])
    pro_measured_00[i] = np.sum(random_values==0)
    pro_measured_01[i] = np.sum(random_values==1)
    pro_measured_10[i] = np.sum(random_values==2)
    pro_measured_11[i] = np.sum(random_values==3)

    pro_measured_00[i] = pro_measured_00[i]/(pro_measured_00[i]+pro_measured_01[i])
    pro_measured_01[i] = 1-pro_measured_00[i]
    pro_measured_10[i] = pro_measured_10[i]/(pro_measured_10[i]+pro_measured_11[i])
    pro_measured_11[i] = 1-pro_measured_10[i]
    if i%10000 == 0:
        print(i,'finished')

df00 = pd.DataFrame(pro_measured_00, columns=['00'])
df01 = pd.DataFrame(pro_measured_01, columns=['01'])
df10 = pd.DataFrame(pro_measured_10, columns=['10'])
df11 = pd.DataFrame(pro_measured_11, columns=['11'])
merged_df_a0 = pd.concat([coordinate, a_0, df00, df01], axis=1)
merged_df_a1 = pd.concat([coordinate, a_1, df10, df11], axis=1)

merged = pd.concat([merged_df_a0,merged_df_a1],axis=0)
merged.to_excel("C:/Users/86136/Desktop/dataset_generation/uncorrelated/unfaked_data_uncorrelated_measured_train.xlsx", index=False)