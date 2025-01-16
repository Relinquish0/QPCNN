import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_propagation
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap

def plotting_PDM(count_nor,times_,ab):
        colors = [(20/255, 54/255, 95/255), ##14365F
                (118/255, 162/255, 135/255), ##76A287
                (248/255, 242/255, 236/255),##F8F2EC
                (191/255, 217/255, 229/255),##BFD9E5
                (214/255, 79/255, 56/255)] ##D64F38
        # Create a custom color map
        cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors)


        fig, ax = plt.subplots(figsize = (14 , 7))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        cax = ax.imshow(count_nor, cmap=cmap_custom, vmin=0, vmax=1)#b

        ax = plt.gca()  # Gets the current axis object
        ax.spines['top'].set_linewidth(4)  # Set the line width of the border to 4
        ax.spines['right'].set_linewidth(4)  
        ax.spines['bottom'].set_linewidth(4)  
        ax.spines['left'].set_linewidth(4)   
        plt.tick_params(axis='both', width=4)

        ax.set_xticks([-0.5,5.5,11.5])  
        ax.set_xticklabels(['-π','0', 'π'],fontsize=52, color='black')

        # Set the scale value of the ordinate
        ax.set_yticks([-0.5,5.5])  
        ax.set_yticklabels(['π','0'],fontsize=52, color='black')
        # colorbar
        
        colorbar = fig.colorbar(cax, shrink=0.8, aspect=20, pad=0.05)
        colorbar.ax.tick_params(labelsize=40,width = 6)  # Set the scale label size
        #plt.title(times)
        plt.xlabel('φ',fontsize=60, fontweight = 'bold',color='black',labelpad = -8)

        plt.ylabel('θ',fontsize=60, fontweight = 'bold',color='black',labelpad = -15)
        #plt.show()
        plt.savefig('C:/Users/86136/Desktop/QP/output/PDM/{}_{}.pdf'.format(times_,ab), dpi=300,bbox_inches='tight')  # PDF格式

def real_experiment(times):
        average_X = 6
        average_Y = 12
        timee = 1
        #X and Y refer to the number of rows and columns in the map, and timee refers to N(=1).

        p_born = pd.read_excel("C:/Users/86136/Desktop/QP/dataset_generation/correlated/data_bell_correlated_output.xlsx")
        p_born = p_born.to_numpy()
        p_born_00 = p_born[:,6]
        p_born_01 = p_born[:,7]
        p_born_10 = p_born[:,8]
        p_born_11 = p_born[:,9]
        #born's rule result

        coordinate = pd.read_excel("C:/Users/86136/Desktop/QP/output/PDM/coordinate.xlsx")
        coordinate = coordinate.to_numpy()
        phi = coordinate[:,14]
        theta = coordinate[:,13]
        #coordinate transformation
        
        f = pd.read_excel('C:/Users/86136/Desktop/QP/output/f/{}.xlsx'.format(times))
        f = f.to_numpy()
        #f(b|axy)        
        f_0 = f[:,0]#Column 0 of 0-40000 is 00, and column 1 of 0-40000 is 01
        f_1 = f[:,1]#Column 0 of 40,000-80000 is 10, and column 1 of 40,000-80000 is 11

        count_final_00 = np.zeros((average_X,average_Y))
        count_final_01 = np.zeros((average_X,average_Y))
        count_final_10 = np.zeros((average_X,average_Y))
        count_final_11 = np.zeros((average_X,average_Y))

        count_final_a_0 = np.zeros((average_X,average_Y))
        count_final_a_1 = np.zeros((average_X,average_Y))

        point = np.zeros((average_X,average_Y))
        count_nor_00 = np.zeros((average_X,average_Y))
        count_nor_01 = np.zeros((average_X,average_Y))
        count_nor_10 = np.zeros((average_X,average_Y))
        count_nor_11 = np.zeros((average_X,average_Y))

        for i in range(0,p_born.shape[0],1):#phi(-pi,pi),theta:(0,pi)
            m = 0
            n = 0
            while np.pi * m / average_X < theta[i]:   #We devide pi into 6 pieces and confirm the position of theta.
                                                #We assume that the accuracy of detector is pi/6
                m+=1
            while 2 * np.pi * n / average_Y - np.pi< phi[i]:   #We devide pi into 12 pieces and confirm the position of theta.
                                                #We assume that the accuracy of detector is pi/12
                n+=1         
            #Perform the above operations to find the cells corresponding to the xy group

            count_particle = np.random.choice([0,1,2,3], size=timee, p=[p_born_00[i], p_born_01[i],p_born_10[i],p_born_11[i]])
            #Born's rule is used to determine the measurement results of this pair of particles

            if count_particle.sum() == 0:
                after_erase = np.random.choice([0,1],size=1,p=[1-f_0[i],f_0[i]])
                count_final_00[m-1,n-1] += after_erase
                count_final_a_0[m-1,n-1] += after_erase
                #If the particle is measured as 00, then after the erase of f, if it remains, it is counted as count_final_00
            elif count_particle.sum() == 1:
                after_erase = np.random.choice([0,1],size=1,p=[1-f_1[i],f_1[i]])
                count_final_01[m-1,n-1] += after_erase
                count_final_a_0[m-1,n-1] += after_erase
                #If the particle is measured as 01, then after the erase of f, if it remains, it is counted as count_final_01
            elif count_particle.sum() == 2:
                after_erase = np.random.choice([0,1],size=1,p=[1-f_0[i+p_born.shape[0]],f_0[i+p_born.shape[0]]])
                count_final_10[m-1,n-1] += after_erase
                count_final_a_1[m-1,n-1] += after_erase
                #If the particle is measured as 10, then after the erase of f, if it remains, it is counted as count_final_10
            elif count_particle.sum() == 3:
                after_erase= np.random.choice([0,1],size=1,p=[1-f_1[i+p_born.shape[0]],f_1[i+p_born.shape[0]]])
                count_final_11[m-1,n-1] += after_erase
                count_final_a_1[m-1,n-1] += after_erase
                #If the particle is measured as 11, then after the erase of f, if it remains, it is counted as count_final_11              

        for m in range(0,average_X,1):
            for n in range(0,average_Y,1):
                # count_nor_sum = count_final_00[m-1,n-1]+count_final_01[m-1,n-1]+count_final_10[m-1,n-1]+count_final_11[m-1,n-1]
                # if count_nor_sum!=0:
                    count_nor_00[m-1,n-1] = count_final_00[m-1,n-1] /count_final_a_0[m-1,n-1]
                    count_nor_01[m-1,n-1] = count_final_01[m-1,n-1] /count_final_a_0[m-1,n-1]
                    count_nor_10[m-1,n-1] = count_final_10[m-1,n-1] /count_final_a_1[m-1,n-1]
                    count_nor_11[m-1,n-1] = count_final_11[m-1,n-1] /count_final_a_1[m-1,n-1]

        #plotting
        plotting_PDM(times_=times,count_nor=count_nor_00,ab='00')
        plotting_PDM(times_=times,count_nor=count_nor_01,ab='01')
        plotting_PDM(times_=times,count_nor=count_nor_10,ab='10')
        plotting_PDM(times_=times,count_nor=count_nor_11,ab='11')
        plt.close('all')#Turn off to prevent too much memory

for time in ['epoch0','epoch0','epoch1','epoch5','epoch20','epoch100','unfaked data']:#'epoch0','epoch0','epoch1','epoch5','epoch20','epoch100','unfaked data'
    #The first time to generate an image don't know why the title and label size is not right so born twice#
    
    real_experiment(times=time)
    
