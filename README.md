This is to serve the paper [Building a human-like observer using deep learning in an extended Wigner's friend experiment](https://arxiv.org/abs/2409.04690)

In **function_lib/pt_generator.py**, one can genearte quantum probability in the circumstance of both uncorrelated measurement and correlated measurement. In **model.py**, one can run QP-CNN for generating $p^n(b|axy)$ in different epoch. In **function_lib/exposeddata_generator.py**, we offer abundant tools to construct the characterization metrics. The whole process in the papar is shown on **data_prep.ipynb**.

The first version of QP-CNN is complished on Aug 21th 2024. We note that Numpy 2.0 is not compatible with original code, while Numpy 2.0 is necessary in Opencv to construct *Morphing Polygon*, *Adverage Shannon Entropy* and *Probability Density Map*. To keep the program working, we degrade Numpy to 1.26.4 when running QP-CNN, and upgrade Numpy to 2.3.2.

Other quantum system is also compatible in QP-CNN. One could revising QPCNN/function_lib/pt_generator to change the quantum state. For example, we consider $\mu = 1$ in the state $\Psi = |0\rangle|0\rangle - \mu|1\rangle|1\rangle$ in the main text. When considering $\mu \neq 1$, we transform the code in line 21, `entangle = np.kron(psi_negative, psi_negative) - 1*np.kron(psi_positive, psi_positive) `, into `entangle = np.kron(psi_negative, psi_negative) - 0.5*np.kron(psi_positive, psi_positive) `. Then continue the process.

Update time: Aug 26th 2025.
