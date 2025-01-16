import pandas as pd
import numpy as np

# 读取Excel文件

def to_npy(file_path):
    # 读取Excel文件
    excel_file_path = "{}.xlsx".format(file_path)
    df = pd.read_excel(excel_file_path)

    # 将DataFrame转换为NumPy数组
    numpy_array = df.to_numpy()

    # 保存为.npy文件
    npy_file_path = "{}.npy".format(file_path)
    np.save(npy_file_path, numpy_array)

def set_file(save_path):
    train_file_path = '{}/Data_train_correlated'.format(save_path)
    to_npy(train_file_path)

    test_file_path = '{}/Data_test_correlated'.format(save_path)
    to_npy(test_file_path)

    infer_file_path = '{}/Data_infer_correlated'.format(save_path)
    to_npy(infer_file_path)

    train_file_path = '{}/Data_train_uncorrelated'.format(save_path)
    to_npy(train_file_path)

    test_file_path = '{}/Data_test_uncorrelated'.format(save_path)
    to_npy(test_file_path)

    infer_file_path = '{}/Data_infer_uncorrelated'.format(save_path)
    to_npy(infer_file_path)
