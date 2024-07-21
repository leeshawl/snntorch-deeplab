import sys
import os

# 将 dataset 目录添加到 Python 搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen


from dataset.augdataset import augvocDataset
from torch.utils.data import DataLoader
import cv2

torch.cuda.empty_cache()


# 定义批次大小和图像转换的均值和标准差
batch_size = 16
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 定义图像和目标的转换器
img_transform = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.ToTensor(),
    transforms.Normalize(*mean_std)
])
tar_transform = transforms.Compose([transforms.Resize([64,64])])
test_tar_transform = transforms.Compose([transforms.Resize([64,64])])

# 加载数据集并创建数据加载器
traindata = augvocDataset("C:/Users/93074/Documents/VOC2012/VOC2012/ImageSets/Segmentation", transform=img_transform, target_transform=tar_transform)
testdata = augvocDataset("C:/Users/93074/Documents/VOC2012/VOC2012/ImageSets/Segmentation", "val", transform=img_transform, target_transform=test_tar_transform)
trainloader = DataLoader(traindata, batch_size, True)
testloader = DataLoader(testdata, batch_size)

# 决定模型运行的设备（GPU或CPU）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cls = torch.zeros((21))
for i,t in tqdm(iter(trainloader)):
     t = t.flatten()
     for k in t:
         cls[k]+=1
med = torch.median(cls).item()
weight = med/cls
torch.save(weight,"weight1.pt")
# 加载已经训练好的权重
weight = torch.load("weight0.pt")


# 定义PoissonGenerator类，用于生成Poisson分布的脉冲
class PoissonGenerator(nn.Module):
    """
    用于生成Poisson脉冲的模块。

    参数:
    - gpu (bool): 是否使用GPU进行计算。
    """

    def __init__(self, gpu=False):
        super().__init__()
        self.gpu = gpu

    def forward(self, inp, rescale_fac=1.0):
        """
        生成Poisson脉冲。

        参数:
        - inp (Tensor): 输入信号。
        - rescale_fac (float): 输入信号的缩放因子。

        返回:
        - Tensor: 生成的Poisson脉冲。
        """
        rand_inp = torch.rand_like(inp).cuda() if self.gpu else torch.rand_like(inp)
        return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

spike_grad = surrogate.atan()
beta=0.99

class deeplab(nn.Module):
    def __init__(self,step):
        super(deeplab,self).__init__()
        self.step = step
        bias_flag = False
        affine_flag = True
        self.input = PoissonGenerator(True)
        self.c1 = nn.Conv2d(3,64,3,1,1,bias=bias_flag)
        self.b1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c2 = nn.Conv2d(64,64,3,1,1,bias=bias_flag)
        self.b2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.a1 = nn.AvgPool2d(3,2,1)
        self.skip1 = nn.Conv2d(64,21,1)

        self.c3 = nn.Conv2d(64,128,3,1,1,bias=bias_flag)
        self.b3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c4 = nn.Conv2d(128,128,3,1,1,bias=bias_flag)
        self.b4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.a2 = nn.AvgPool2d(3,2,1)
        self.skip2 = nn.Conv2d(128,21,1)

        self.c5 = nn.Conv2d(128,256,3,1,1,bias=bias_flag)
        self.b5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c6 = nn.Conv2d(256,256,3,1,1,bias=bias_flag)
        self.b6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c7 = nn.Conv2d(256,256,3,1,1,bias=bias_flag)
        self.b7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif7 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.a3 = nn.AvgPool2d(3,2,1)
        self.skip3 = nn.Conv2d(256,21,1)

        self.c8 = nn.Conv2d(256,1024,3,1,1,bias=bias_flag)
        self.b8 = nn.ModuleList([nn.BatchNorm2d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif8 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c9 = nn.Conv2d(1024,1024,3,1,1,bias=bias_flag)
        self.b9 = nn.ModuleList([nn.BatchNorm2d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif9 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.skip4 = nn.Conv2d(1024,21,1)
        self.up1 = nn.Sequential(
                    nn.ConvTranspose2d(21, 21, kernel_size=3, stride=2, bias=False),
                    nn.UpsamplingBilinear2d((16, 16)))
        self.lif10 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.up2 = nn.Sequential(
                    nn.ConvTranspose2d(21, 21, kernel_size=3, stride=2, bias=False),
                    nn.UpsamplingBilinear2d((32, 32)))
        self.lif11 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.up3 = nn.Sequential(
                    nn.ConvTranspose2d(21, 21, kernel_size=3, stride=2, bias=False),
                    nn.UpsamplingBilinear2d((64, 64)))

    def forward(self,x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()
        mem8 = self.lif8.init_leaky()
        mem9 = self.lif9.init_leaky()
        mem10 = self.lif10.init_leaky()
        mem11 = self.lif11.init_leaky()
        # mem12 = self.lif12.init_leaky()
        # mem9 = self.lif9.init_leaky()
        spk_out = []
        for step in range(self.step):
            cur1 = self.input(x)
            cur1 = self.c1(cur1)
            cur1 = self.b1[step](cur1)
            spk1, mem1 = self.lif1(cur1,mem1)
            cur2 = self.c2(spk1)
            cur2 = self.b2[step](cur2)
            spk2, mem2 = self.lif2(cur2,mem2)
            skip1 = self.skip1(spk2)
            spk2 = self.a1(spk2)

            cur3 = self.c3(spk2)
            cur3 = self.b3[step](cur3)
            spk3, mem3 = self.lif3(cur3,mem3)
            cur4 = self.c4(spk3)
            cur4 = self.b4[step](cur4)
            spk4, mem4 = self.lif4(cur4,mem4)
            skip2 = self.skip2(spk4)
            spk4 = self.a2(spk4)

            cur5 = self.c5(spk4)
            cur5 = self.b5[step](cur5)
            spk5, mem5 = self.lif5(cur5,mem5)
            cur6 = self.c6(spk5)
            cur6 = self.b6[step](cur6)
            spk6, mem6 = self.lif6(cur6,mem6)
            cur7 = self.c7(spk6)
            cur7 = self.b7[step](cur7)
            spk7, mem7 = self.lif7(cur7,mem7)
            skip3 = self.skip3(spk7)
            spk7 = self.a3(spk7)

            cur8 = self.c8(spk7)
            cur8 = self.b8[step](cur8)
            spk8, mem8 = self.lif8(cur8,mem8)
            cur9 = self.c9(spk8)
            cur9 = self.b9[step](cur9)
            spk9, mem9 = self.lif9(cur9,mem9)
            skip4 = self.skip4(spk9)

            cur10 = self.up1(skip4) + skip3
            spk10,mem10 = self.lif10(cur10,mem10)
            cur11 = self.up2(spk10) + skip2
            spk11,mem11 = self.lif11(cur11,mem11)
            cur12 = self.up3(spk11) + skip1

            spk_out.append(cur12)
        spk_out = torch.stack(spk_out).mean(0)
        return spk_out

# Define the number of epochs for training
num_epochs = 200
# Initialize the number of iterations and the step size for printing loss information
num_iters = 0
num_step = 20
# Initialize the network and send it to the specified device
net = deeplab(num_step).to(device)
# Initialize the optimizer and specify the learning rate, whether to use the AMSGrad algorithm, and the weight decay parameter
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, amsgrad=True, weight_decay=5e-4)
# Initialize the loss function, ignoring index 255 and using weighted loss
loss_fn = nn.CrossEntropyLoss(ignore_index=255,weight=weight).to(device)
# Define the milestones for the learning rate scheduler
milestones = [int(milestone*num_epochs) for milestone in [0.5, 0.8]]
# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# Initialize the maximum IoU for monitoring model performance
maxiou = 0

# Main training loop
# training loop
for epoch in range(1,num_epochs+1):
    # Set the network to training mode
    net.train()
    # Initialize the list to store the loss history of this epoch
    loss_hist = []
    # Iterate through the training data
    for i, (datas, targets) in enumerate(tqdm(trainloader)):
        # Send the target to the specified device
        target = targets.long().to(device)
        # Forward pass
        spk_rec = net(datas.to(device))
        # Calculate the loss
        loss = loss_fn(spk_rec,target)
        # Clear the gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the network parameters
        optimizer.step()
        # Store the loss value
        # Store loss history for future plotting
        loss_hist.append(loss.item())

    # Print the average loss of this epoch and write it to the file
    print("epoch",epoch,":",np.array(loss_hist).mean())
    with open('loss.txt','a') as f:
        f.write("epoch"+str(epoch)+":"+str(np.array(loss_hist).mean())+'\n')
    # Update the learning rate
    scheduler.step()
    # Clear the GPU cache to save memory
    torch.cuda.empty_cache()
    # If the average loss of this epoch is greater than 2, skip the following evaluation steps
    if np.array(loss_hist).mean() > 2:
        continue

    # Initialize the confusion matrix for evaluating model performance
    cm = np.zeros((21,21))
    # Set the network to evaluation mode
    with torch.no_grad():
        net.eval()
        # Iterate through the test data
        for i, (datas, targets) in enumerate(tqdm(testloader)):
            # Forward pass
            spk_rec = net(datas.to(device)) #T X B X C X H X W
            # Get the predicted class index
            _,idxs = spk_rec.max(1)
            # Send the predicted and true class indices to the CPU for subsequent calculation
            idx = idxs.detach().cpu().flatten().int().numpy()
            target = targets.flatten().int().numpy()
            # Update the confusion matrix
            for i,t in enumerate(target):
                if t != 255:
                    cm[t,idx[i]] += 1

    # Calculate the mean IoU
    MIoU = np.diag(cm) / (
                    np.sum(cm, axis=1) +
                    np.sum(cm, axis=0) -
                    np.diag(cm)+1e-15)
    # Print and record the mean IoU of the current epoch
    print(np.mean(MIoU[1:]))
    MIoU = np.mean(MIoU)
    # If the current mean IoU is greater than the previously recorded maximum IoU, update the maximum IoU and save the model parameters
    if MIoU > maxiou:
        maxiou = MIoU
        torch.save(net.state_dict(),"model.pth")
    # Print and record the current mean IoU
    print("miou:"+str(MIoU))
    with open('miou.txt','a') as f:
        f.write(str(MIoU)+'\n')
