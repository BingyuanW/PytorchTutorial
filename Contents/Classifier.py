
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


import matplotlib.pyplot as plt
import numpy as np

# ###################################################################
# 加载数据
# ###################################################################
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 转换数据格式

trainset = torchvision.datasets.CIFAR10(root='D:\Python_code\PytorchLearn\myProject\Classifier', train=True, download=False,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='D:\Python_code\PytorchLearn\myProject\Classifier',train=False, download=False,
                                        transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img/2 + 0.5
    npimg =img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ###################################################################
# 定义卷积神经网络
# ###################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 3个输入通道，6个输出通道，窗口大小为5x5
        self.pool = nn.MaxPool2d(2, 2)   # 最大池化， 窗口大小为2x2
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射操作，y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # LeNet-5 的参数：120，84，10
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu((self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net = net.to(device)

# ###################################################################
# 定义损失函数和优化器
# ###################################################################
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降法，学习率0.01，动量0.9

# ###################################################################
# 训练网络
# ###################################################################
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()  # 清零参数的梯度

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f ' %
                  (epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0
print('Finished Training')

"""
# ###################################################################
# 保存网络
# ###################################################################
# PATH = 'D:\Python_code\PytorchLearn\myProject\Classifier\cifar_net.pth'
# torch.save(net.state_dict(), PATH)


'''
# 图片可视化
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

# ###################################################################
# 用测试集测试网络
# ###################################################################
dataiter = iter(testloader)
images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 加载保存的网络
# net = Net()
# net.load_state_dict(torch.load(PATH))

outputs = net(images)
correct = 0
total = 0

with torch.no_grad():   # 不被追踪（track），不计算梯度
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


"""





