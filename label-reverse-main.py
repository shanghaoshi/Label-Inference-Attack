import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import SGD
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import Module
from torch import nn
import matplotlib.pyplot as plt
from cifarclassifier import Classifier
import matplotlib.pyplot as plt


def get_real_dist(labels, batch_size):
    dist_vector=torch.zeros(10)
    for i in range(batch_size):
        for j in range(10):
            if labels[i]==j:
                dist_vector[j]=dist_vector[j]+1
    return dist_vector
    

def attack(gradient, batch_size, level):

    original_vector=torch.sum(gradient,dim=1)
    base_value, index=torch.topk(original_vector,3)
    medium_vector=original_vector-torch.sum(base_value)/3
#    activation_vector=medium_vector
    activation_vector=medium_vector+(1-level)/(level+1e-4)*torch.sum(medium_vector)/20
    distribution=torch.zeros(10)
    estimated_dist=torch.zeros(10)
    
    summation=0
    for i in range(10):
        if activation_vector[i]>0:
            distribution[i]=0
        else:
            distribution[i]=-activation_vector[i]
            summation=summation-activation_vector[i]
    distribution=distribution/summation
#    print(distribution)
    for i in range(10):
        estimated_dist[i]=int(distribution[i]*batch_size)
        if distribution[i]>0 and estimated_dist[i]==0:
            estimated_dist[i]=1
    remain=int(batch_size-torch.sum(estimated_dist))
#    print(remain)
    if remain>0:
        index=torch.randint(low=0, high=10, size=(remain,))
        for i in range(remain):
            estimated_dist[index[i]]=estimated_dist[index[i]]+1
    
    return estimated_dist

def data_split(train_data, train_label, ratio, selected_list, select_num, client_num):
    train_data=torch.from_numpy(train_data)
    train_label=torch.Tensor(train_label)
    data_num=train_data.shape[0]
    data_size=int(50000/client_num)
    images=torch.zeros(data_size, 32, 32, 3)
    labels=torch.zeros(data_size)
    index=torch.randint(low=0, high=int(50000*(1-ratio)), size=(int(data_size*(1-ratio)),)) 
    
    images[0:int(data_size*(1-ratio)),:,:,:]=train_data[index,:,:,:]
    labels[0:int(data_size*(1-ratio))]=train_label[index]
    
    acc=0
    for selected_label in selected_list:
        selected_label_index=(train_label[int(data_size*(1-ratio)):-1]==selected_label).nonzero(as_tuple=True)[0]+int(data_size*(1-ratio))
        images[int(data_size*(1-ratio))+acc:int(data_size*(1-ratio))+int(data_size*ratio/select_num)+acc,:,:,:]=train_data[selected_label_index[acc:int(data_size*ratio/select_num)+acc],:,:,:]
        labels[int(data_size*(1-ratio))+acc:int(data_size*(1-ratio))+int(data_size*ratio/select_num)+acc]=train_label[selected_label_index[acc:int(data_size*ratio/select_num)+acc]]
        acc=acc+int(data_size*ratio/select_num)
    return images, labels

def visualization(estimated_dist, real_dist):
    print(real_dist)
    print(estimated_dist)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.bar(['1','2','3','4','5','6','7','8','9','10'], real_dist, color="blue")
    plt.xlabel("Real Distribution", fontsize=16)
    plt.ylabel("Sample Number", fontsize=16)
    plt.xticks(fontsize=13)
    plt.ylim(0,400)
    plt.yticks([50, 100, 150, 200, 250, 300, 350], fontsize=13)
    plt.subplot(1, 2, 2)
    plt.bar(['1','2','3','4','5','6','7','8','9','10'], estimated_dist, color="red")
    plt.xticks(fontsize=13)
    plt.yticks([], fontsize=13)
    plt.ylim(0,400)
    plt.xlabel("Estimated Distribution", fontsize=16)
    plt.savefig("results/distribution.pdf")
    
    

if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    batch_size = 1200
    select_num=2
    test_rounds= 1
    client_num = 50
    noniid_level=0.4
    selected_class=[3,5]
    guess=0.4
    all_epoch = 1
    transform = transforms.Compose([transforms.ToTensor(),])
    train_data = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transform)
#    train_data = torchvision.datasets.FashionMNIST(root='./train', train=True, transform=ToTensor(), download=True)
#    train_data = torchvision.datasets.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
    images, labels=data_split(train_data.data, train_data.targets, noniid_level, selected_class, select_num, client_num)
    correct_num=0
    for k in range(test_rounds):
        index=torch.randint(low=0, high=int(50000/client_num), size=(batch_size,))
        model = Classifier().to(device)
        sgd = SGD(model.parameters(), lr=1e-2)
        loss_fn = CrossEntropyLoss()
        gradient=torch.zeros(10,512).to(device)
        model.fc3.weight.data = 1/(512)*torch.ones(10, 512).to(device).clone().detach()
       
        for current_epoch in range(all_epoch):
            model.train()
            train_x = images.view(int(50000/client_num),3,32,32)[index].to(device)
            train_label = labels[index].to(device)
            predict_y = model(train_x.float())
            sgd.zero_grad()
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()
        gradient=-(model.fc3.weight.data-1/(512)*torch.ones(10, 512).to(device))/5
        estimated_dist=attack(gradient.cpu(), batch_size, guess)
        real_dist=get_real_dist(labels[index].to(device), batch_size)
         
        a_hat=estimated_dist>200
        a_real=real_dist>200
        correct_num=correct_num+len(torch.eq(a_hat, a_real).nonzero())
        torch.cuda.empty_cache()
#        visualization(estimated_dist, real_dist)    
#        print("Real Distribution:", real_dist)
#        print("Estimated Distribution:", estimated_dist)
    print("Accuracy:", correct_num/(test_rounds*10))
    print("Finished!")
    
    
    



            

    
    
    
        
