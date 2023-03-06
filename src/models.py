from torch import nn
import torch.nn.functional as F
from Sagnet.modules.sag_resnet import sag_resnet
from Sagnet.modules.loss import *
from Sagnet.modules.utils import *

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Sagnet(nn.Module):
    def __init__(self, args):
        global model
        model = sag_resnet(depth=int(args.depth),
                           pretrained=not args.from_sketch,
                           num_classes=args.num_classes,
                           drop=args.drop,
                           sagnet=args.sagnet,
                           style_stage=args.style_stage)

        print(model)
        model = torch.nn.DataParallel(model).cuda()