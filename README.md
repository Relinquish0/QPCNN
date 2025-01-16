This is to serve the paper [Building a human-like observer using deep learning in an extended Wigner's friend experiment](https://arxiv.org/abs/2409.04690)

The whole procedure is divided into three steps——dataset generation, QPCNN and output. It is worth noting that these different codes correspond to different steps, so there are some minor changes that need to be made in different processes, all of which are mentioned in the comments
In addition, each piece of code needs to change the corresponding address to make sure it works correctly.

Lenovo pro16 2021 Xiaoxin

Python 3.11.5

Pytorch 2.0.1

```markdown
DriverVersion    Name
30.0.13002.7003  AMD Radeon(TM) Graphics
31.0.15.3699     NVIDIA GeForce GTX 1650
```

Update time: Jan 16th 2025.
A more simplified way to run it is being written...
# dataset generation
Here is the data generation section, divided into Correlated parts and Uncorrelated parts.
Correlated parts is used to generate the dataset for training phase and output phase
Uncorrelated parts is used to generate the dataset for pre-training phase.

The first step is to open the bell - program (thanks to Mr. Chen Jixuan's code) to generate Born's rule results.
Then open the corresponding notepad and copy and paste it into excel.
Note that the code is in write mode, so the planner must be cleared before running the code.

Change your name to something you recognize, and change the address in your code. Here with me


```markdown

| data_bell_uncorrelated_train      | The training set for the pretraning phase |
| data_bell_uncorrelated_validation | The test set for the pretraining phase    |
| data_bell_correlated_train        | The training set for the traning phase    |
| data_bell_correlated_validation   | The test set for the training phase       |
| data_bell_correlated_output       | The output set for the output phase       |
```
Then open their respective data_preset runs, and note the comments in the code. "measured" means the measured result $p^m$ after they have done N=100,000 tests.

# QPCNN
First, the main processing file, put the pm in the dataset generation as the training set, the test set, the deduction set

One of the most difficult things to do is unfaked data correlated files.
The first six lines are coordinates, the seventh line is a(0 or 1), and the eighty-ninth line is $p^t$(a=0) and $p^t$(a=1), which are the results of born's rule and need to be copied and pasted by hand *2
The eleventh line is $p^m$. Just copy unfaked_data_correlated_measured_output and insert $p^t$ between $a$ and $p^m$

Then there is the training section, the parameters are already adjusted, there is a self.save path, change this parameter to the neural network in the file can be done
Run the pre-train first. For 200,000 pieces of data it would take about two hours

Rerun infer
Open save in outputdata, look at the name of the pretraining file, and change the model load path to the model saved by pretraining.
For 200,000 data sets, 80,000 inferences would take about 11 hours. (Computer Lenovo Xiaoxin, GTX1650, this program can only be run with gpu)

The final inference results are included in outputdata inference, $p^n$ is output.

The excel "error" contains the error and entropy of each epoch, the entropy is $p^o$ when N->∞.
You can't use this directly, so you have to do it in the last output file

# output

This step is used after the neural network, first open the f_calculation f and po(n=100,000), be careful to change the corresponding address.
$p^n$ can be obtained directly from neural networks, and pt and pm can be obtained on unfaked_data_correlated neural networks.
For unfaked data, you had to faked a special column set to f=1 to run it.
Note that f, $p^o$, ASE are 80,000 rows (a,x,y), but the measurement in the output phase is 40000 groups (a, b|x,y),
You can also calculate the result of born's rule directly as epoch 101 and then change the name, which is the method I used at the time
At the end, it outputs f, po and ASE(entropy).
The process takes about 40 minutes

Then open the NN polytopes file to select the epoch you want to run directly, you can change different alpha to get different Outlines
Then import the resulting excel directly into origin and you can draw pictures

Finally, PDM, this step is more troublesome. The first step is to figure out the relative coordinates of each instance, in column 1314
In coordinate, for example, you can also change the formula depending on how you map
Then just run the program and the epoch you want, paying attention to normalization.
