
import torch
import torch.nn.functional as F
import torch.nn as nn

class CNNCifar10(nn.Module):  
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
class cifar_6766(torch.nn.Module):
    
    def __init__(self):
        super(cifar_6766, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3,2,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,4,kernel_size=3,stride=1,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*4,32),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = x.view(-1, 7*7*4)
        x = self.dense(x)
        return x


class cifar_405(torch.nn.Module):#88acc
    
    def __init__(self):
        super(cifar_405, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False),
                                         torch.nn.LeakyReLU())
        self.conv2 = torch.nn.Sequential(
                                         torch.nn.Conv2d(1,1,kernel_size=3,stride=2,padding=0,bias=False),
                                         torch.nn.LeakyReLU(),
                                         torch.nn.Conv2d(1,1,kernel_size=3,stride=2,padding=0,bias=False),
                                        #  torch.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0,bias=False),
                                         )
        self.dense = torch.nn.Sequential(
                                        torch.nn.Linear(36,10,bias=False),
                                        #  torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                        #  torch.nn.Linear(32, 10,bias=False)
                                        )
    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        # pdb.set_trace()
        x = self.conv2(x)
        # pdb.set_trace()
        x = x.view(bs, -1)
        x = self.dense(x)
        return x

class cifar_502(torch.nn.Module): #90acc
    
    def __init__(self):
        super(cifar_502, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3,2,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,4,kernel_size=3,stride=2,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(3*3*4,10),
                                        #  torch.nn.ReLU(),
                                        # #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                        #  torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3*3*4)
        x = self.dense(x)
        return x

class cifar_1138(torch.nn.Module): #94acc
    
    def __init__(self):
        super(cifar_1138, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3,4,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(4,8,kernel_size=3,stride=2,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(3*3*8,10),
                                        #  torch.nn.ReLU(),
                                        # #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                        #  torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3*3*8)
        x = self.dense(x)
        return x



class cifar_13314(torch.nn.Module): #97acc
    
    def __init__(self):
        super(cifar_13314, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3,4,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(4,8,kernel_size=3,stride=1,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*8,32),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*8)
        x = self.dense(x)
        return x

# class cifar_13354(torch.nn.Module): #96acc
    
#     def __init__(self):
#         super(cifar_13354, self).__init__()
#         self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=1,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.MaxPool2d(stride=2,kernel_size=2))
#         self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*4,64),
#                                          torch.nn.ReLU(),
#                                         #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
#                                          torch.nn.Linear(64, 10)
#                                         )
#         # import pdb; pdb.set_trace()
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(-1, 7*7*4)
#         x = self.dense(x)
#         return x

# class cifar_19664(torch.nn.Module): #96acc
    
#     def __init__(self):
#         super(cifar_19664, self).__init__()
#         self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=1,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(2,6,kernel_size=3,stride=2,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.MaxPool2d(stride=2,kernel_size=2))
#         self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*6,64),
#                                          torch.nn.ReLU(),
#                                         #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
#                                          torch.nn.Linear(64, 10)
#                                         )
#         # import pdb; pdb.set_trace()
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(-1, 7*7*6)
#         x = self.dense(x)
#         return x

# class cifar_25974(torch.nn.Module): #98acc
    
#     def __init__(self):
#         super(cifar_25974, self).__init__()
#         self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=1,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(2,8,kernel_size=3,stride=2,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.MaxPool2d(stride=2,kernel_size=2))
#         self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*8,64),
#                                          torch.nn.ReLU(),
#                                         #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
#                                          torch.nn.Linear(64, 10)
#                                         )
#         # import pdb; pdb.set_trace()
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(-1, 7*7*8)
#         x = self.dense(x)
#         return x


# class Lenet(torch.nn.Module):
#     def __init__(self):
#         super(Lenet, self).__init__()
#         self.conv1 =torch.nn.Conv2d(1, 6, 5)
#         self.relu1 =torch.nn.ReLU()
#         self.pool1 =torch.nn.MaxPool2d(2)
#         self.conv2 =torch.nn.Conv2d(6, 16, 5)
#         self.relu2 =torch.nn.ReLU()
#         self.pool2 =torch.nn.MaxPool2d(2)
#         self.fc1 =torch.nn.Linear(256, 120)
#         self.relu3 =torch.nn.ReLU()
#         self.fc2 =torch.nn.Linear(120, 84)
#         self.relu4 =torch.nn.ReLU()
#         self.fc3 =torch.nn.Linear(84, 10)
#         self.relu5 =torch.nn.ReLU()

#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.relu1(y)
#         y = self.pool1(y)
#         y = self.conv2(y)
#         y = self.relu2(y)
#         y = self.pool2(y)
#         y = y.view(y.shape[0], -1)
#         y = self.fc1(y)
#         y = self.relu3(y)
#         y = self.fc2(y)
#         y = self.relu4(y)
#         y = self.fc3(y)
#         # y = self.relu5(y)
#         return y


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes,track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes,track_running_stats=False)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion*planes,track_running_stats=False)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes,track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes,track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes,track_running_stats=True)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes,track_running_stats=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.st = 't'

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, track_running_stats=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = torch.nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        # if self.st == 'test':
        #     # import pdb; pdb.set_trace()
        #     # # print('conv1 out')
        #     # print(out[:3,:3,:3,:3])
        #     self.st = 't'
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_feature(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_feature, self).__init__()
        self.in_planes = 64
        self.st = 't'

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, track_running_stats=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = torch.nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        # if self.st == 'test':
        #     # import pdb; pdb.set_trace()
        #     # # print('conv1 out')
        #     # print(out[:3,:3,:3,:3])
        #     self.st = 't'
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return feature, out
    def fc_forward(self, x):
        out = self.linear(x)
        return out

class fc(torch.nn.Module):
    def __init__(self, block, num_classes=10):
        super(fc, self).__init__()
        self.linear = torch.nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out



def classifier(num_classes):
    return fc(BasicBlock,num_classes)


def ResNet18_cifar10():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18_cifar10_feature():
    return ResNet_feature(BasicBlock, [2, 2, 2, 2])

def ResNet50_cifar10():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet50_cifar10_feature():
    return ResNet_feature(Bottleneck, [3, 4, 6, 3])

''' ConvNet '''
class ConvNet_cifar10(torch.nn.Module):
    def __init__(self, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', channel=3, num_classes=10, im_size = (28,28)):
        super(ConvNet_cifar10, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = torch.nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return torch.nn.Sigmoid()
        elif net_act == 'relu':
            return torch.nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return torch.nn.LeakyReLU(negative_slope=0.01)


    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return torch.nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return torch.nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return torch.nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return torch.nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return torch.nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [torch.nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return torch.nn.Sequential(*layers), shape_feat