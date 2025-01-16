This step is used after the neural network, first open the f_calculation f and po(n=100,000), be careful to change the corresponding address
pn can be obtained directly from neural networks, and pt and pm can be obtained on unfaked_data_correlated neural networks
For unfaked data, you had to faked a special column set to f=1 to run it,
Note that f, po, ASE are 80,000 rows (a,x,y), but the measurement in the output phase is 40000 groups (a, b|x,y),
You can also calculate the result of born's rule directly as epoch 101 and then change the name, which is the method I used at the time
At the end, it outputs f, po and ASE(entropy).
The process takes about 40 minutes

Then open the NN polytopes file to select the epoch you want to run directly, you can change different alpha to get different Outlines
Then import the resulting excel directly into origin and you can draw pictures

Finally, PDM, this step is more troublesome. The first step is to figure out the relative coordinates of each instance, in column 1314
In coordinate, for example, you can also change the formula depending on how you map
Then just run the program and the epoch you want, paying attention to normalization.

这一步是用在神经网络之后的，首先打开f_calculation计算f和po(n=100,000)，注意改成相应的地址
pn直接从神经网络得到，pt和pm在神经网络中的unfaked_data_correlated可以得到
至于unfaked data需要特别拿出来一列设为f=1，接着进行运算，
注意f、po、ASE是80，000行（a,x,y)，但是output phase中的measurement是40000组（a，b|x，y）,
也可以直接把born's rule的结果当成epoch 101来计算后面改名字就好，我当时用的方法是后者
最后会输出f，po和ASE(entropy)
这一过程大概要跑40分钟

接着打开NN polytopes文件选取想要的epoch直接运行即可，可以改不同的alpha得到不同的轮廓
然后把得到的excel直接迁入origin就可以画图了

最后是PDM,这一步比较麻烦。首先要先算出各个instance的相对坐标，在第1314列
例子在coordinate中，也可以根据不同的映射方式改变公式
然后直接运行程序和想要的epoch就可以了，注意归一化。