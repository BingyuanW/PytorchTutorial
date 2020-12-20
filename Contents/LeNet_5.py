
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)  # 1个输入通道，6个输出通道，窗口大小为3x3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 仿射操作，y=Wx+b
        self.fc1 = nn.Linear(16*6*6, 120)  # LeNet-5 的参数：120，84，10
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 最大池化， 窗口大小为2x2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # 改变x的列数和行数 ？？？
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
'''
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
'''

input = torch.randn(1, 1, 32, 32)  # 可以理解为 1张图，1个通道，高宽为32x32 ？？？
output = net(input)
# print(output)
'''
tensor([[ 0.0703, -0.0037, -0.0505, -0.0741,  0.0734, -0.0062, -0.0246,  0.0369,
         -0.0968,  0.0878]], grad_fn=<AddmmBackward>)
'''
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
# print(loss)
'''
tensor(1.7817, grad_fn=<MseLossBackward>)
'''
# print(loss.grad_fn)
'''
<MseLossBackward object at 0x0000021C4D3556A0>
'''
# print(loss.grad_fn.next_functions[0][0])
'''
<AddmmBackward object at 0x0000021C55B0E080>
'''
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
'''
<AccumulateGrad object at 0x0000021C4D3556A0>
'''
net.zero_grad()
# print('conv1 .bias.grad before backward')
# print(net.conv1.bias.grad)
'''
conv1 .bias.grad before backward
None
'''

loss.backward()
# print('conv1 .bias.grad after backward')
# print(net.conv1.bias.grad)
'''
conv1 .bias.grad after backward
tensor([-0.0002,  0.0097, -0.0080,  0.0061, -0.0009, -0.0001])
'''
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


