The paper is called QP-CNN
First, the main processing file, put the pm in the dataset generation as the training set, the test set, the deduction set

One of the most difficult things to do is unfaked data correlated files.
The first six lines are coordinates, the seventh line is a(0 or 1), and the eighty-ninth line is pt0 and pt1, which are the results of born's rule and need to be copied and pasted by hand *2
The eleventh line is pm
Just copy unfaked_data_correlated_measured_output and insert pt between a and pm

Then there is the training section, the parameters are already adjusted, there is a self.save path, change this parameter to the neural network in the file can be done
Run the pre-train first
For 200,000 pieces of data it would take about two hours

Rerun infer
Open save in outputdata, look at the name of the pretraining file, and change the model load path to the model saved by pretraining
For 200,000 data sets, 80,000 inferences would take about 11 hours

Computer Lenovo Xiaoxin, graphics card 1650, this program can only be run with gpu

The final inference results are included in outputdata inference, pn is output.
error This excel contains the error and entropy of each epoch, the entropy is po when N->∞,
You can't use this directly, so you have to do it in the last output file


论文中名为QP-CNN
先是最主要的处理文件，把dataset generation里面的pm放进来作为训练集，测试集，推演集

最麻烦的一件事情是unfaked data correlated文件（output）
前六行是坐标，第七行是a(0 or 1), 第八九行是pt0和pt1，就是born's rule的结果，需要手动复制粘贴过来*2
第十十一行是pm
其实就是unfaked_data_correlated_measured_output复制过来，a和pm中间空两行插入pt罢了

然后就是训练部分，参数都已经调好了，里面有一个self.save path,把这个参数改成神经网络所在的文件就可以了
先运行pre-train
如果是200，000条数据的话大概要两个小时

再运行infer
打开outputdata里面的save，看看pretraining文件的名称，将model load path改成pretraining保存的模型
如果是200，000条数据，80，000个推演集大概要11个小时

电脑联想小新，显卡1650，这个程序只有有gpu才可以运行

最后的推演结果在outputdata inference里面，输出的是pn。
error这个excel里面有每个epoch的误差和熵，这个熵是po当N->∞的时候的值，
这个不能直接用，所以还是要在最后一步output文件中导出来算