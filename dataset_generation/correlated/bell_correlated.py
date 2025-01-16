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
    entangle = np.kron(psi_negative, psi_negative) - np.kron(psi_positive, psi_positive)  # An entangled matrix of two particles
    prob = np.dot(entangle.T, np.dot(np.kron(p1, p2), entangle)) / 2
    probability = np.real(prob[0][0])
    return probability


f = open("C:/Users/86136/Desktop/dataset_generation/correlated/data_bell_correlated.txt", "a", encoding="UTF-8")

n = int(input("Random generation frequency："))
f.write(f"{'xa'} {'ya'} {'za'} {'xb'} {'yb'} {'zb'} {'pa_0'} {'pa_1'} {'pb_0'} {'pb_1'}\n") 

for num in range(n):
    # Random generated directiona
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    xa = np.sin(theta) * np.cos(phi)
    ya = np.sin(theta) * np.sin(phi)
    za = np.cos(theta)

    # Random generated directionb
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    xb = np.sin(theta) * np.cos(phi)
    yb = np.sin(theta) * np.sin(phi)
    zb = np.cos(theta)

    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    xc = np.sin(theta) * np.cos(phi)
    yc = np.sin(theta) * np.sin(phi)
    zc = np.cos(theta)

    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    xd = np.sin(theta) * np.cos(phi)
    yd = np.sin(theta) * np.sin(phi)
    zd = np.cos(theta)
    for i in range(1,2,1):
        if i == 1:#A0B0
            x1 = xa
            y1 = ya
            z1 = za
            x2 = xb
            y2 = yb
            z2 = zb
        if i == 2:#A0B1
            x1 = xa
            y1 = ya
            z1 = za
            x2 = xd
            y2 = yd
            z2 = zd
        if i == 3:#A1B0
            x1 = xc
            y1 = yc
            z1 = zc
            x2 = xb
            y2 = yb
            z2 = zb     
        if i == 4:#A1B1
            x1 = xc
            y1 = yc
            z1 = zc
            x2 = xd
            y2 = yd
            z2 = zd   
        Sa1 = (I + (x1 * Jx + y1 * Jy + z1 * Jz)) / 2  # Measure the projection matrix of particle 1 as 1 in direction a
        Sa0 = (I - (x1 * Jx + y1 * Jy + z1 * Jz)) / 2  # Measure the projection matrix of particle 1 as 0 in direction a
        Sb1 = (I + (x2 * Jx + y2 * Jy + z2 * Jz)) / 2  # Measure the projection matrix of particle 2 as 1 in direction b
        Sb0 = (I - (x2 * Jx + y2 * Jy + z2 * Jz)) / 2  # Measure the projection matrix of particle 2 as 0 in direction b
        p1_ = cal(Sa1, I)  # Just measure the probability that particle 1 will result in 1
        p11 = cal(Sa1, Sb1)  # Measure the probability that particles 1 and 2 will both result in 1
        p10 = cal(Sa1, Sb0)  # The probability that measuring particles 1 and 2 results in 1, 0, respectively
        p0_ = cal(Sa0, I)  # Only measure the probability that particle 1 will result in 0
        p01 = cal(Sa0, Sb1)  # Measure the probability that particles 1 and 2 result in 0,1, respectively
        p00 = cal(Sa0, Sb0)  # Measure the probability that particles 1 and 2 will both result in 0
        f.write(f"{x1} {y1} {z1} {x2} {y2} {z2} {p00} {p01} {p10} {p11}\n")         


f.close()
