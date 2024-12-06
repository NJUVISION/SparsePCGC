import torch
import MinkowskiEngine as ME


class InceptionResNet(torch.nn.Module):
    """Inception Residual Network
    """
    
    def __init__(self, channels, kernel_size=3, dimension=3):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=dimension)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x

        return out

######################### ResNet #########################
class ResNet(torch.nn.Module):   
    def __init__(self, channels, kernel_size=3, dimension=3):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out

######################### Dilated ResNet #########################
class DilatedResNet(torch.nn.Module):
    """Dilated Residual Network
    """
    def __init__(self, channels, kernel_size=3, dilation=2, dimension=3):
        super().__init__()
        dilation_list = [1, 2, 3, 5, 7, 11][:dilation]
        self.conv0_list = torch.nn.ModuleList([ME.MinkowskiConvolution(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dilation=dilation, 
            bias=True,  
            dimension=dimension) for dilation in dilation_list])
        self.linear0 = ME.MinkowskiConvolution(
            in_channels=channels*len(dilation_list),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=dimension)
        #
        self.conv1_list = torch.nn.ModuleList([ME.MinkowskiConvolution(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dilation=dilation, 
            bias=True,  
            dimension=dimension) for dilation in dilation_list])
        self.linear1 = ME.MinkowskiConvolution(
            in_channels=channels*len(dilation_list),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=dimension)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out = ME.cat([conv(x) for conv in self.conv0_list])
        out = self.relu(self.linear0(out))
        out = ME.cat([conv(out) for conv in self.conv1_list])
        out = self.linear1(out)
        out += x

        return out

# def make_layer(block, block_layers, channels, dimension=3):
#     """make stacked layers.
#     """
#     layers = []
#     for i in range(block_layers):
#         layers.append(block(channels=channels, dimension=dimension))
        
#     return torch.nn.Sequential(*layers)


class ResNetBlock(torch.nn.Module):
    def __init__(self, channels=32, kernel_size=3, block_layers=3, dimension=3, block_type='inception'):
        super().__init__()
        if block_type=='resnet': Net = ResNet
        if block_type=='inception': Net = InceptionResNet
        if block_type=='dilation': Net = DilatedResNet
        self.layers = torch.nn.ModuleList()
        for i in range(block_layers):
            self.layers.append(Net(channels=channels, kernel_size=kernel_size, dimension=dimension))

    def forward(self, x):
        out = x
        for resnet in self.layers:
            out = resnet(out)
        if len(self.layers)>1:
            out += x
        return out