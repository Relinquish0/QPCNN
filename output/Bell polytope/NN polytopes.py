import cv2
import pandas as pd
import numpy as np
import alphashape
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from shapely.geometry import Polygon


def roundness_cal(times):
    excel_file_path = 'C:/Users/86136/Desktop/output/Bell polytope/xy_{}_exterior.xlsx'.format(times)
    df1 = pd.read_excel(excel_file_path)
    data = df1.to_numpy()  # Suppose each row has two features: x and y
    points = data[:, :2]  # Take the first two columns as the set of points
    points = points.astype(np.float32)  # Convert data type

    if points.shape[0] < 3:
        raise ValueError("Point set must contain at least three points to form a convex hull.")

    # Computational convex hull
    hull = cv2.convexHull(points)

    # Calculate the convex hull area
    hull_area = cv2.contourArea(hull)

    # Calculate the minimum circumscribed circle
    (x, y), radius = cv2.minEnclosingCircle(points)
    roundness = hull_area / (np.pi * radius * radius)
    # print(f"{time}Convex Hull Area: {hull_area}")
    # print(f"{time}Center of the Minimum Enclosing Circle: ({x}, {y})")
    # print(f"{time}Radius of the Minimum Enclosing Circle: {radius}")
    print(f"{times}roundness: {roundness}")

def set_2dimensional_slice(times):
    excel_file_path = "C:/Users/86136/Desktop/output/p_o/{}.xlsx".format(times)
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
    data_PLOT.to_excel("C:/Users/86136/Desktop/output/Bell polytope/xy_{}.xlsx".format(times),index=False)
    #========================================extract the outline==========================================================
    df=pd.read_excel("C:/Users/86136/Desktop/output/Bell polytope/xy_{}.xlsx".format(times))
    points = df.to_numpy()
    #Alpha Shape
    alpha_shape = alphashape.alphashape(points, 1)  # alpha Parameters control the fineness of the shape    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(points[:, 0], points[:, 1], color='red')
    print(alpha_shape.geom_type)
    # Plotting Alpha Shape
    if isinstance(alpha_shape, Polygon):
        x, y = alpha_shape.exterior.xy
        ax.plot(x, y, color='blue')
        exterior_coords = list(alpha_shape.exterior.coords)
        exterior_coords = np.array(exterior_coords)
        #print(exterior_coords)
    # plt.xlim(-3,3)
    # plt.ylim(-3,3)
    #plt.show()


    def residuals(r, x, y):
        return (x**2 + y**2 - r**2)
    initial_guess = [2.8]
    x = exterior_coords[:,0]
    y = exterior_coords[:,1]
    res = least_squares(residuals, initial_guess, args=(x, y))
    # The fitting result of least square method was extracted
    r_fit = np.abs(res.x[0])  # 

    print(times,"radius",r_fit)
    exterior_coords = pd.DataFrame(exterior_coords)
    exterior_coords.to_excel("C:/Users/86136/Desktop/output/Bell polytope/xy_{}_exterior.xlsx".format(times),index=False)
#set_infer()

for time in ['epoch0','epoch1','epoch2','epoch5','epoch10','epoch20','epoch100','unfaked data']:#
    set_2dimensional_slice(times=time)
    roundness_cal(times= time)
    
