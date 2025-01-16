import numpy as np
import pandas as pd
pd.set_option( 'display.precision',10)

def calculating_f():
    for times in range(0,101,1):#'epoch0','epoch1','epoch2','epoch10','epoch100','unfaked data'
        p = pd.read_excel("C:/Users/86136/Desktop/QPCNN/unfaked_data_nonlocal.xlsx")
        p = p.to_numpy()#recieve theta in unfaked data
        pt_0 = p[:,7]
        pt_1 = p[:,8]
        pm_0 = p[:,9]
        pm_1 = p[:,10]
        p_n = pd.read_excel("C:/Users/86136/Desktop/QPCNN/output_Data/QPCNN_output/inference/output_iter0_epoch{}.xlsx".format(times))
        p_n = p_n.to_numpy()    
        pn_0 = p_n[:,8]
        pn_1 = p_n[:,9]

        f = np.zeros([pm_0.shape[0],2])
        for i in range(0,pm_0.shape[0],1):
            #set b as 0, b' as 1
            f[i,1] = 1
            j = pn_1[i]*pt_0[i]
            j = np.clip(j, 1e-10, 1 - 1e-10)
            k = pt_1[i]*pn_0[i]
            k = np.clip(k, 1e-10, 1 - 1e-10)
            f[i,0] = j/k
            if f[i,0] >= f[i,1]:
                f[i,0] = 1
                f[i,1] = k/j        
        f = pd.DataFrame(f)
        f.to_excel('C:/Users/86136/Desktop/QP/output/f/epoch{}.xlsx'.format(times), index=False)

def calculating_p_o():
    count_a = pd.read_excel("C:/Users/86136/Desktop/QP/dataset_generation/correlated/unfaked_data_correlated_measured_output_count.xlsx")
    count_a = count_a.to_numpy()
    #Obtaining the measurement result in output phase
    count_0_0 = count_a[:,6]
    count_0_1 = count_a[:,7]
    count_1_0 = count_a[:,8]
    count_1_1 = count_a[:,9]
    count_0_0 = count_0_0.astype(int)   
    count_0_1 = count_0_1.astype(int)  
    count_1_0 = count_1_0.astype(int)
    count_1_1 = count_1_1.astype(int)   

    coordinate = pd.read_excel("C:/Users/86136/Desktop/QPCNN/unfaked_data_nonlocal.xlsx")
    coordinate = coordinate.to_numpy()    
    coordinate_0 = coordinate[:40000,:7] 
    coordinate_1 = coordinate[40000:,:7] 
    for times in range(3,101,1):

        f = pd.read_excel('C:/Users/86136/Desktop/QP/output/f/epoch{}.xlsx'.format(times))
        f = f.to_numpy()
        
        p_o_a_0 = np.zeros([count_0_0.shape[0],2])
        p_o_a_1 = np.zeros([count_0_1.shape[0],2])
        for i in range(0,count_0_0.shape[0],1):
            count_b_0 = np.clip(count_b_0, 1e-10, 1 - 1e-10)
            count_b_1 = np.clip(count_b_1, 1e-10, 1 - 1e-10)
            p_o_a_0[i,0] = count_b_0.sum()/(count_b_0.sum()+count_b_1.sum())
            p_o_a_0[i,1] = count_b_1.sum()/(count_b_0.sum()+count_b_1.sum())
        for i in range(0,count_1_0.shape[0],1):
            count_b_0 = np.random.choice([0, 1], size = count_1_0[i], p=[1-f[i+count_1_0.shape[0],0], f[i+count_1_0.shape[0],0]])
            count_b_1 = np.random.choice([0, 1], size = count_1_1[i], p=[1-f[i+count_1_0.shape[0],1], f[i+count_1_0.shape[0],1]])
            count_b_0 = np.clip(count_b_0, 1e-10, 1 - 1e-10)
            count_b_1 = np.clip(count_b_1, 1e-10, 1 - 1e-10)
            p_o_a_1[i,0] = count_b_0.sum()/(count_b_0.sum()+count_b_1.sum())
            p_o_a_1[i,1] = count_b_1.sum()/(count_b_0.sum()+count_b_1.sum())
        
        p_o_a_0 = np.concatenate((coordinate_0,p_o_a_0),axis=1)
        #p_o_a_0 = pd.DataFrame(p_o_a_0)
        p_o_a_1 = np.concatenate((coordinate_1,p_o_a_1),axis=1)
        #p_o_a_1 = pd.DataFrame(p_o_a_1)
        p_o = np.concatenate((p_o_a_0,p_o_a_1),axis=0)
        p_o = pd.DataFrame(p_o)
        p_o.to_excel('C:/Users/86136/Desktop/QP/output/p_o/epoch{}.xlsx'.format(times), index=False)

def calculating_entropy():
    entropy_list = np.zeros([101,1])
    for times in range(0,101,1):
        po = pd.read_excel('C:/Users/86136/Desktop/QP/output/p_o/epoch{}.xlsx'.format(times))
        po = po.to_numpy()
        po = po[:,7]
        entropy = 0
        for i in range(0,po.shape[0],1):
            po[i] = np.clip(po[i], 1e-10, 1 - 1e-10)
            entropy = entropy-po[i]*np.log(po[i])-(1-po[i])*np.log(1-po[i])
        entropy_list[times] = entropy/po.shape[0]
        print(times,':',entropy/po.shape[0])
    
    entropy_list = pd.DataFrame(entropy_list)
    entropy_list.to_excel('C:/Users/86136/Desktop/QP/output/Shannon entropy.xlsx'.format(times), index=False)


calculating_f() 
calculating_p_o()
calculating_entropy()

    