import torch
import torch.nn as nn

class CNNBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        nn.Conv2d(3, 32, )
        self.encoders_in = [3,32,64,128,256]
        self.encoders_out = [32,64,128,256,512]
        self.conv = []
        self.deconv = []
        
        for i in range(len(self.encoders_in)):
            self.conv.append(nn.Conv2d(self.encoders_in[i], self.encoders_out[i], kernel_size=3, stride=2, padding=1, dilation=1))
            self.conv.append(nn.BatchNorm2d(self.encoders_out[i]))
            
        self.relu = nn.ReLU(inplace=True)
        
        self.decoders_in = [512,256,128,64]
        self.decoders_out = [256,128,64,32]
        
        for i in range(len(self.decoders_in)):
            self.deconv.append(nn.ConvTranspose2d(self.decoders_in[i], self.decoders_out[i], kernel_size=3, stride=2, padding=1, dilation=1))
            self.deconv.append(nn.BatchNorm2d(self.decoders_out[i]))
            
        self.classifier = nn.Conv2d(32, 5, kernel_size=1)

        
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        
    def forward(self, x):
        
        for i in range(len(self.encoders_in)):
            x = self.conv[i](x)
                
            if not isinstance(self.conv[i], nn.BatchNorm2d):
                x = self.relu(x)
        
        # x1 = self.bnd1(self.relu(self.conv1(x)))
        
        y = x # transition from conv to deconv
        
        for i in range(len(self.decoders_in)):
            y = self.deconv[i](y)
                
            if not isinstance(self.deconv[i], nn.BatchNorm2d):
                y = self.relu(y)

        score = self.classifier(y)
        
        return score