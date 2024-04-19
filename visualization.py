import numpy as np
import matplotlib.pyplot as plt
import matplotlib

non_iid=[0.2,0.4,0.6,0.8,1.0]
mnist_rate=[0.89, 1.0,1.0,1.0,1.0]
fmnist_rate=[0.94, 1.0,1.0,1.0,1.0]
cifar_rate=[0.96, 1.0,1.0,1.0,1.0]

plt.figure(figsize=(7,6))
plt.plot(non_iid, mnist_rate, marker="^", color="blue")
plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=18)
plt.ylim(0.5,1.1)
plt.xlabel("Non-iid Degree", fontsize=18)
plt.ylabel("Inferring Accuracy", fontsize=18)
plt.savefig("results/Inferring Accuracy MNIST.png")

plt.figure(figsize=(7,6))
plt.plot(non_iid, fmnist_rate, marker="^", color="green")
plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=18)
plt.ylim(0.5,1.1)
plt.xlabel("Non-iid Degree", fontsize=18)
plt.ylabel("Inferring Accuracy", fontsize=18)
plt.savefig("results/Inferring Accuracy FMNIST.png")

plt.figure(figsize=(7,6))
plt.plot(non_iid, cifar_rate, marker="^", color="red")
plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=18)
plt.ylim(0.5,1.1)
plt.xlabel("Non-iid Degree", fontsize=18)
plt.ylabel("Inferring Accuracy", fontsize=18)
plt.savefig("results/Inferring Accuracy CIFAR.png")
