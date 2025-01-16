import torch
import torch.nn
import numpy as np
'''
函数torch.permute：改变维度
    举例：x=torch.randn(2,3,4)
         x.permute(0,2,1)
         结果(2,3,4)->(2,4,3)

函数torch.view:
'''

# 输入数据x的尺寸变化过程形如: (39601, 4, 64) -> (64, 4, 39601) -> (1, 64, 4, 39601)升维
def rect34(x):
    x = x.permute(2, 1, 0)  # 第0维与第2维的数据进行位置交换，类似于transpose
    x = x.view(1, x.shape[0], x.shape[1], x.shape[2])  # tensor reshape
    return x


# 输入数据x的尺寸变化过程形如: (1, 64, 4, 19900) -> (64, 4, 19900) -> (19900, 4, 64)
def rect43(x):
    x = x.view(x.shape[1], x.shape[2], x.shape[3])  # tensor reshape
    x = x.permute(2, 1, 0)  # 第0维与第2维的数据进行位置交换，类似于transpose
    return x


# 输入数据x的尺寸变化过程形如: (1, 3, 4, 19900) -> (1, 3, 4, 199, 199)。
# 在尺寸为(200,200)的全零数组中，以直线y=x为对称轴，将每组的19900个数据点对称放置在其中(分为左下三角形区域和右上三角形区域)
def dia_symmetry(x):
    mould199_199 = torch.zeros((x.shape[0], x.shape[1], x.shape[2], 199, 199))
    # 横轴为c3轴，纵轴为c5轴
    index = 0
    for col in range(0, 199):  # 0-198列
        for row in range(col + 1):  # 0-(col-1)行，共col行
            mould199_199[:, :, :, col, row] = x[:, :, :, index]  # 左下三角形区域
            mould199_199[:, :, :, row, col] = x[:, :, :, index]  # 右上三角形区域
            index += 1
    return mould199_199


# 输入数据x的尺寸变化过程形如: (1, 1, 4, 199, 199) -> (1, 1, 4, 19900)
# 在尺寸为(200,200)的关于直线y=x对称的数组中，将该直线一侧(三角形区域内)的19900个数据点提取出来
def dia_asymmetry(x):
    mould19900 = torch.zeros((x.shape[0], x.shape[1], x.shape[2], 19900))
    index = 0
    for col in range(0, 199):  # 0-198列
        for row in range(col + 1):  # 0-(col-1)行，共col行
            mould19900[:, :, :, index] = x[:, :, :, row, col]  # 提取右上角三角形
            index += 1
    return mould19900


# 数据预处理[输入的x即为ED数据，其尺寸为(19900, 10, 3)]
# 输出尺寸为(19900, depth, 2)的(c3,c5)数组与尺寸为(19900, depth, 3)的d_k^{n=begin~(begin+depth)}数组
def data_prep(x, begin, depth):
    # c预处理
    c = x[:, 2:4, 0]  # 获取(c3, c5)坐标，尺寸为(19900, 2)
    c = c.view(c.shape[0], 1, c.shape[1])  # 尺寸变为(19900, 1, 2)
    c = c.repeat(1, depth, 1)  # 沿第1维的方向复制数组depth次，数据尺寸变为(19900, depth, 2)
    c = rect34(c)  # 尺寸变为(1, 2, depth, 19900)
    c = dia_symmetry(c)  # 尺寸变为(1, 2, depth, 199, 199)
    c = c.view(c.shape[0], c.shape[1], c.shape[2], c.shape[3] * c.shape[4])  # 尺寸变为(1, 2, depth, 199*199)
    c = rect43(c)  # 尺寸变为(199*199, depth, 2)

    # dk预处理
    dk = x[:, begin:begin + depth, 0:3]  # 获取d_k^{n=begin~(begin+depth)}数组，尺寸为(19900, depth, 3)
    dk = rect34(dk)  # 尺寸变为(1, 3, depth, 19900)
    dk = dia_symmetry(dk)  # 尺寸变为(1, 3, depth, 199, 199)
    dk = dk.view(dk.shape[0], dk.shape[1], dk.shape[2], dk.shape[3] * dk.shape[4])  # 尺寸变为(1, 3, depth, 199*199)
    dk = rect43(dk)  # 尺寸变为(199*199, depth, 3)
    return c, dk


# 数据提取，其中的数据尺寸变化过程如下，将从data_prep得到的199*199个数据点提取出对应于原始ED文件中相同(c3,c5)坐标区域内的19900个数据点
def data_extract(x):  # 设输入x的尺寸为(199*199, 4, 3)
    x = rect34(x)  # (1, 3, 4, 199*199)
    x = x.view(x.shape[0], x.shape[1], x.shape[2], 199, 199)  # (1, 3, 4, 199, 199)
    x = dia_asymmetry(x)  # (1, 3, 4, 19900)
    x = rect43(x)  # (19900, 4, 3)
    return x
