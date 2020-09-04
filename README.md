# Multiception
Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy

A new convolution type to boost the performance of depthwise seperable convolution (DSConv)

![](/others/multiception.png)

```python

class Multiception(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernels):
        super(Multiception, self).__init__()
        padding_dict = {1:0, 3:1, 5:2, 7:3}
        self.seps = nn.ModuleList()
        for kernel in kernels:
            sep = nn.Conv2d(in_channel,in_channel, kernel_size = kernel,stride =1,padding = padding_dict[kernel],dilation=1,groups=in_channel, bias=False)
            self.seps.append(sep)
        self.bn1 = nn.BatchNorm2d(in_channel*len(kernels)) 
        self.pointwise = nn.Conv2d(in_channel*len(kernels), out_channel, 1, stride, 0, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)       

    def forward(self, x):
        seps = []
        for sep in self.seps:
            seps.append(sep(x))
        out_seq = torch.cat(seps, dim=1)
        out = self.pointwise(self.bn1(out_seq))
        out = self.bn2(out)
        return out 
```
  
# Citation
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy", 16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

School of Computer Science, The University of Sydney

# Experiments

## Performance comparison with depthwise seperable convolution (DSConv)

#### Cifar 10
![](/others/multiception-vs-dsconv1.png)

#### Cifar 100
![](/others/multiception-vs-dsconv2.png)

#### STL 10
![](/others/multiception-vs-dsconv3.png)

## Performance comparison with standard convolution
![](/others/multiception-vs-standard.png)

