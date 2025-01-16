import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
f = pd.read_excel("C:/Users/86136/Desktop/dataset_generation/correlated/data_bell_correlated_validation.xlsx")#train/validation/output
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

N = 100000#measurement rounds
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
    if i%10000 == 0:
        print(i,'finished')

df00 = pd.DataFrame(pro_measured_00, columns=['00'])
df01 = pd.DataFrame(pro_measured_01, columns=['01'])
df10 = pd.DataFrame(pro_measured_10, columns=['10'])
df11 = pd.DataFrame(pro_measured_11, columns=['11'])

merged_df_a0 = pd.concat([coordinate, a_0, df00, df01], axis=1)
merged_df_a1 = pd.concat([coordinate, a_1, df10, df11], axis=1)

merged = pd.concat([merged_df_a0,merged_df_a1],axis=0)
merged.to_excel("C:/Users/86136/Desktop/dataset_generation/correlated/unfaked_data_correlated_measured_validation.xlsx", index=False)

#For the output inference set, the following comments need to be unlocked to get the results measured by Alice and Bob in the output phase
# gf00 = pd.DataFrame(times_measured_00, columns=['time00'])
# gf01 = pd.DataFrame(times_measured_01, columns=['time01'])
# gf10 = pd.DataFrame(times_measured_10, columns=['time10'])
# gf11 = pd.DataFrame(times_measured_11, columns=['time11'])

# merge = pd.concat([coordinate,gf00,gf01,gf10,gf11],axis=1)
# merge.to_excel("C:/Users/86136/Desktop/dataset_generation/correlated/unfaked_data_correlated_measured_output_count.xlsx", index=False)
