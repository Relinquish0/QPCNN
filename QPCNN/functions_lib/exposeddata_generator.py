import numpy as np
import pandas as pd
import os
import alphashape
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from shapely.geometry import Polygon
import cv2
from scipy.ndimage import binary_propagation
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap

pd.set_option( 'display.precision',10)

def calculating_f(load_path):
    # Folder creation
    os.makedirs('{}/exposed_data/f'.format(load_path), exist_ok=True)
    os.makedirs('{}/exposed_data/po'.format(load_path), exist_ok=True)
    os.makedirs('{}/exposed_data/polytopes'.format(load_path), exist_ok=True)

    # Extracting p^t into correct form
    pt = pd.read_excel("{}/Dataset/pt_correlated_output.xlsx".format(load_path))
    pt = pt.to_numpy()#recieve theta in unfaked data
    pt_trans = np.zeros((2*pt.shape[0],2))
    for i in range(pt.shape[0]):
        pt_trans[i,0] = pt[i,6]/(pt[i,6]+pt[i,7])
        pt_trans[i,1] = pt[i,7]/(pt[i,6]+pt[i,7])
        pt_trans[i+pt.shape[0],0] = pt[i,8]/(pt[i,8]+pt[i,9])
        pt_trans[i+pt.shape[0],1] = pt[i,9]/(pt[i,8]+pt[i,9])
    pt_0 = pt_trans[:,0]
    pt_1 = pt_trans[:,1]

    # Extracting p^m
    pm = pd.read_excel("{}/Dataset/pm_correlated_output.xlsx".format(load_path))
    pm = pm.to_numpy()#recieve theta in unfaked data
    pm_0 = pm[:,7]
    pm_1 = pm[:,8]

    for times in range(0,101,1):#'epoch0','epoch1','epoch2','epoch10','epoch100','unfaked data'
        p_n = pd.read_excel("{}/output_Data/QPCNN_output/inference/output_iter0_epoch{}.xlsx".format(load_path,times))
        p_n = p_n.to_numpy()    
        pn_0 = p_n[:,8]
        pn_1 = p_n[:,9]

        f = np.zeros([pm_0.shape[0],2])
        for i in range(0,pm_0.shape[0],1):
            #设b为0，b'为1
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
        f.to_excel('{}/exposed_data/f/epoch{}.xlsx'.format(load_path,times), index=False)

    # Repeating this procedure again and set epoch 101 as unfaked data, when f = 1
    f = np.ones([pm_0.shape[0],2])
    f = pd.DataFrame(f)
    f.to_excel('{}/exposed_data/f/epoch101.xlsx'.format(load_path), index=False)

def calculating_p_o(load_path):
    count_a = pd.read_excel("{}/Dataset/pm_correlated_output_count.xlsx".format(load_path))
    count_a = count_a.to_numpy()

    count_0_0 = count_a[:,6]#00的次数
    count_0_1 = count_a[:,7]#01的次数
    count_1_0 = count_a[:,8]#10的次数
    count_1_1 = count_a[:,9]#11的次数
    count_0_0 = count_0_0.astype(int)   #声明为整数数组
    count_0_1 = count_0_1.astype(int)  
    count_1_0 = count_1_0.astype(int)   #声明为整数数组
    count_1_1 = count_1_1.astype(int)   

    coordinate = pd.read_excel("{}/Dataset/pm_correlated_output.xlsx".format(load_path))
    coordinate = coordinate.to_numpy()    
    coordinate_0 = coordinate[:40000,:7] 
    coordinate_1 = coordinate[40000:,:7] 
    for times in range(0,102,1):#'epoch0','epoch1','epoch2','epoch10','epoch100','unfaked data'
        f = pd.read_excel('{}/exposed_data/f/epoch{}.xlsx'.format(load_path,times))
        f = f.to_numpy()
        
        p_o_a_0 = np.zeros([count_0_0.shape[0],2])
        p_o_a_1 = np.zeros([count_0_1.shape[0],2])
        for i in range(0,count_0_0.shape[0],1):
            count_b_0 = np.random.choice([0, 1], size = count_0_0[i], p=[1-f[i,0], f[i,0]]) # based on probability
            count_b_1 = np.random.choice([0, 1], size = count_0_1[i], p=[1-f[i,1], f[i,1]])
            #=======================================================================
            # count_b_0 = np.around(count_0_0[i]*f[i,0]) # based on fraction
            # count_b_1 = np.around(count_0_1[i]*f[i,1])
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
        p_o.to_excel('{}/exposed_data/po/epoch{}.xlsx'.format(load_path,times), index=False)

def calculating_entropy(load_path):
    entropy_list = np.zeros([101,1])
    for times in range(0,101,1):
        po = pd.read_excel('{}/exposed_data/po/epoch{}.xlsx'.format(load_path,times))
        po = po.to_numpy()
        po = po[:,7] # 00的次数
        entropy = 0
        for i in range(0,po.shape[0],1):
            po[i] = np.clip(po[i], 1e-10, 1 - 1e-10)
            entropy = entropy-po[i]*np.log(po[i])-(1-po[i])*np.log(1-po[i])
        entropy_list[times] = entropy/po.shape[0]
        print(times,':',entropy/po.shape[0])
    
    entropy_list = pd.DataFrame(entropy_list)
    entropy_list.to_excel('{}/exposed_data/Shannon entropy.xlsx'.format(load_path,times), index=False)

def polytopes(load_path,times):
    excel_file_path = "{}/exposed_data/po/epoch{}.xlsx".format(load_path,times)
    #excel_file_path = 'C:/Users/86136/Desktop/SFWML_simplified/preset_infer_simple.xlsx'
    df1 = pd.read_excel(excel_file_path)
    data = df1.to_numpy()
    data = data[:,6:]
    x_ = []
    y_ = []
    for row in range(0,data.shape[0],4):
        if data[row,0] == 0:
            x = -(data[row,1]-data[row,2]) + (data[row+1,1]-data[row+1,2]) + (data[row+2,1]-data[row+2,2]) + (data[row+3,1]-data[row+3,2])
            y = +(data[row,1]-data[row,2]) + (data[row+1,1]-data[row+1,2]) + (data[row+2,1]-data[row+2,2]) - (data[row+3,1]-data[row+3,2])
            x_.append(x)
            y_.append(y)
        elif data[row,0] == 1:
            x = -(data[row,2]-data[row,1]) + (data[row+1,2]-data[row+1,1]) + (data[row+2,2]-data[row+2,1]) + (data[row+3,2]-data[row+3,1])
            y = +(data[row,2]-data[row,1]) + (data[row+1,2]-data[row+1,1]) + (data[row+2,2]-data[row+2,1]) - (data[row+3,2]-data[row+3,1])
            x_.append(x)
            y_.append(y)
    x_ = np.array(x_)
    y_ = np.array(y_)
    r_ = 0

    data_PLOT = {'x':x_,'y':y_}
    data_PLOT = pd.DataFrame(data_PLOT)
    data_PLOT.to_excel("{}/exposed_data/polytopes/xy_{}.xlsx".format(load_path,times),index=False)

    df=pd.read_excel("{}/exposed_data/polytopes/xy_{}.xlsx".format(load_path,times))

    # 示例数据
    points = df.to_numpy()

    # 生成 Alpha Shape
    alpha_shape = alphashape.alphashape(points, 1)  # alpha 

    # 绘制点
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(points[:, 0], points[:, 1], color='red')
    print(alpha_shape.geom_type)
    # 绘制 Alpha Shape
    if isinstance(alpha_shape, Polygon):
        x, y = alpha_shape.exterior.xy
        ax.plot(x, y, color='blue')
        exterior_coords = list(alpha_shape.exterior.coords)
        exterior_coords = np.array(exterior_coords)
        #print(exterior_coords)
    # plt.xlim(-3,3)
    # plt.ylim(-3,3)
    # 显示结果
    #plt.show()

    def residuals(r, x, y):
        return (x**2 + y**2 - r**2)
    initial_guess = [2.8]
    x = exterior_coords[:,0]
    y = exterior_coords[:,1]
    res = least_squares(residuals, initial_guess, args=(x, y))
    # 
    r_fit = np.abs(res.x[0])  # 取绝对值确保半径为正

    # 
    print(times,"拟合半径",r_fit)
    exterior_coords = pd.DataFrame(exterior_coords)
    exterior_coords.to_excel("{}/exposed_data/polytopes/xy_{}_exterior.xlsx".format(load_path,times),index=False)

def roundness_cal(load_path, times):
    excel_file_path = "{}/exposed_data/polytopes/xy_{}.xlsx".format(load_path,times)
    df1 = pd.read_excel(excel_file_path)
    data = df1.to_numpy()  # 
    points = data[:, :2]  # 
    points = points.astype(np.float32)  # 

    if points.shape[0] < 3:
        raise ValueError("Point set must contain at least three points to form a convex hull.")

    hull = cv2.convexHull(points)

    hull_area = cv2.contourArea(hull)

    (x, y), radius = cv2.minEnclosingCircle(points)

    roundness = hull_area / (np.pi * radius * radius)

    print(f"{times}roundness: {roundness}")

def plotting_PDM(count_nor,times_,ab,output_file):
        colors = [(20/255, 54/255, 95/255), #14365F
                (118/255, 162/255, 135/255), #76A287
                (248/255, 242/255, 236/255),#F8F2EC
                (191/255, 217/255, 229/255),#BFD9E5
                (214/255, 79/255, 56/255)] #D64F38
        # 创建自定义颜色映射
        cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors)


        fig, ax = plt.subplots(figsize = (14 , 7))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        cax = ax.imshow(count_nor, cmap=cmap_custom, vmin=0, vmax=1)#b

        ax = plt.gca()  # 
        ax.spines['top'].set_linewidth(4)  # 
        ax.spines['right'].set_linewidth(4)  # 
        ax.spines['bottom'].set_linewidth(4)  # 
        ax.spines['left'].set_linewidth(4)   # 
        plt.tick_params(axis='both', width=4)

        ax.set_xticks([-0.5,5.5,11.5])  
        ax.set_xticklabels(['-π','0', 'π'],fontsize=52, color='black')

        # 
        ax.set_yticks([-0.5,5.5])  
        ax.set_yticklabels(['π','0'],fontsize=52, color='black')
        # colorbar
        
        colorbar = fig.colorbar(cax, shrink=0.8, aspect=20, pad=0.05)
        colorbar.ax.tick_params(labelsize=40,width = 6)  # 
        # 
        #plt.title(times)
        plt.xlabel('φ',fontsize=60, fontweight = 'bold',color='black',labelpad = -8)

        plt.ylabel('θ',fontsize=60, fontweight = 'bold',color='black',labelpad = -15)
        
        # 
        #plt.show()
        plt.savefig('{}/exposed_data/PDM/epoch{}_{}.pdf'.format(output_file,times_,ab), dpi=300,bbox_inches='tight')  # PDF格式

def real_experiment(file_path,times):
        average_X = 6
        average_Y = 12
        timee = 1
        #timee: N(=1)

        p_born = pd.read_excel("{}/Dataset/pt_correlated_output.xlsx".format(file_path))
        p_born = p_born.to_numpy()
        p_born_00 = p_born[:,6]
        p_born_01 = p_born[:,7]
        p_born_10 = p_born[:,8]
        p_born_11 = p_born[:,9]
        #born's rule      

        phi = np.zeros(p_born.shape[0])
        theta = np.zeros(p_born.shape[0])
        for i in range(p_born.shape[0]):
            ksi = p_born[i,3] - p_born[i,0]
            ita = p_born[i,4] - p_born[i,1]
            zeta = p_born[i,5] - p_born[i,2]
            theta[i] = np.acos(zeta/np.sqrt(ksi*ksi+ita*ita+zeta*zeta))
            phi[i] = np.arctan2(ita,ksi)

        f = pd.read_excel('{}/exposed_data/f/epoch{}.xlsx'.format(file_path,times))
        f = f.to_numpy()
        #f(b|axy)        
        f_0 = f[:,0]# 0-40000:00; 0-40000:01
        f_1 = f[:,1]# 40000-80000:10; 40000-80000:11

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
            while np.pi * m / average_X < theta[i]:   #We devide pi into 360 pieces and confirm the position of theta.
                                                #We assume that the accuracy of detector is pi/360
                m+=1
            while 2 * np.pi * n / average_Y - np.pi< phi[i]:   #We devide pi into 360 pieces and confirm the position of theta.
                                                #We assume that the accuracy of detector is pi/360
                n+=1         
            # finding corresponding cell of xy  

            count_particle = np.random.choice([0,1,2,3], size=timee, p=[p_born_00[i], p_born_01[i],p_born_10[i],p_born_11[i]])
            #deciding the measurement result

            if count_particle.sum() == 0:
                after_erase = np.random.choice([0,1],size=1,p=[1-f_0[i],f_0[i]])
                count_final_00[m-1,n-1] += after_erase
                count_final_a_0[m-1,n-1] += after_erase
                # count_final_00 after erasure
            elif count_particle.sum() == 1:
                after_erase = np.random.choice([0,1],size=1,p=[1-f_1[i],f_1[i]])
                count_final_01[m-1,n-1] += after_erase
                count_final_a_0[m-1,n-1] += after_erase
                # count_final_01 after erasure
            elif count_particle.sum() == 2:
                after_erase = np.random.choice([0,1],size=1,p=[1-f_0[i+p_born.shape[0]],f_0[i+p_born.shape[0]]])
                count_final_10[m-1,n-1] += after_erase
                count_final_a_1[m-1,n-1] += after_erase
                # count_final_10 after erasure
            elif count_particle.sum() == 3:
                after_erase= np.random.choice([0,1],size=1,p=[1-f_1[i+p_born.shape[0]],f_1[i+p_born.shape[0]]])
                count_final_11[m-1,n-1] += after_erase
                count_final_a_1[m-1,n-1] += after_erase
                # count_final_11 after erasure               

        for m in range(0,average_X,1):
            for n in range(0,average_Y,1):
                # count_nor_sum = count_final_00[m-1,n-1]+count_final_01[m-1,n-1]+count_final_10[m-1,n-1]+count_final_11[m-1,n-1]
                # if count_nor_sum!=0:
                    count_nor_00[m-1,n-1] = count_final_00[m-1,n-1] /count_final_a_0[m-1,n-1]
                    count_nor_01[m-1,n-1] = count_final_01[m-1,n-1] /count_final_a_0[m-1,n-1]
                    count_nor_10[m-1,n-1] = count_final_10[m-1,n-1] /count_final_a_1[m-1,n-1]
                    count_nor_11[m-1,n-1] = count_final_11[m-1,n-1] /count_final_a_1[m-1,n-1]

        # Plotting
        plotting_PDM(times_=times,count_nor=count_nor_00,ab='00',output_file=file_path)
        plotting_PDM(times_=times,count_nor=count_nor_01,ab='01',output_file=file_path)
        plotting_PDM(times_=times,count_nor=count_nor_10,ab='10',output_file=file_path)
        plotting_PDM(times_=times,count_nor=count_nor_11,ab='11',output_file=file_path)
        plt.close('all') #

