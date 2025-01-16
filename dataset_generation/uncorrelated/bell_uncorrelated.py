import numpy as np

psi_positive = np.array([[1], [0]])
psi_negative = np.array([[0], [1]])
Jx = np.array([[0, 1], [1, 0]])
Jy = np.array([[0, -1j], [1j, 0]])
Jz = np.array([[1, 0], [0, -1]])
I = np.eye(2)


def cal(p1, p2):
    """
    Calculate the probability of a projection result
    :param p1: Projection matrix of the first particle
    :param p2: Projection matrix of the second particle
    :return: indicates the probability
    """
    entangle = np.kron(psi_negative, psi_negative) - np.kron(psi_positive, psi_positive)  
    prob = np.dot(entangle.T, np.dot(np.kron(p1, p2), entangle)) / 2
    probability = np.real(prob[0][0])
    return probability


f = open("C:/Users/86136/Desktop/dataset_generation/uncorrelated/data_bell_uncorrelated.txt", "a", encoding="UTF-8")

n = int(input("Random generation frequency："))
f.write(f"{'xa'} {'ya'} {'za'} {'xb'} {'yb'} {'zb'} {'pa_0'} {'pa_1'} {'pb_0'} {'pb_1'}\n") 
for x in range(n):
    # Random generation frequencya
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    xa = np.sin(theta) * np.cos(phi)
    ya = np.sin(theta) * np.sin(phi)
    za = np.cos(theta)

    Sa1 = (I + (xa * Jx + ya * Jy + za * Jz)) / 2  # Measure the projection matrix of particle 1 as 1 in direction a
    Sa0 = (I - (xa * Jx + ya * Jy + za * Jz)) / 2  # Measure the projection matrix of particle 1 as 0 in direction a

    p1_ = cal(Sa1, I)  # Just measure the probability that particle 1 will result in 1
    p0_ = cal(Sa0, I)  # Just measure the probability that particle 1 will result in 0
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    xb = np.sin(theta) * np.cos(phi)
    yb = np.sin(theta) * np.sin(phi)
    zb = np.cos(theta)

    Sb1 = (I + (xb * Jx + yb * Jy + zb * Jz)) / 2  # Measure the projection matrix of particle 2 as 1 in direction a
    Sb0 = (I - (xb * Jx + yb * Jy + zb * Jz)) / 2  # Measure the projection matrix of particle 2 as 0 in direction a

    p_1 = cal(I, Sb1)  # Just measure the probability that particle 2 will result in 1
    p_0 = cal(I, Sb0)  # Just measure the probability that particle 2 will result in 0
    f.write(f"{xa} {ya} {za} {xb} {yb} {zb} {p0_} {p1_} {p_0} {p_1}\n")    

f.close()
