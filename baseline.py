import torch
import torch.nn as nn

class CNNBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #nn.Conv2d(3, 32, )
        self.encoders_in = [3,32,64,128,256]
        self.encoders_out = [32,64,128,256,512]
            
        self.relu = nn.ReLU(inplace=True)
        
        self.decoders_in = [512,256,128,64]
        self.decoders_out = [256,128,64,32]
            
        self.conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True))
            for in_channels, out_channels in zip(self.encoders_in, self.encoders_out)
        ])
        
        self.deconv = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True))
            for in_channels, out_channels in zip(self.decoders_in, self.decoders_out)
        ])
            
        self.classifier = nn.Linear(32, 5)


        
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        
    def forward(self, x):
        # Pass through encoder layers
        for layer in self.conv:
            x = layer(x)
        
        # Pass through decoder layers
        for layer in self.deconv:
            x = layer(x)
        
        # Flatten the output for the classifier
        # x = torch.flatten(x, 1)
        
        # Final classification layer
        score = self.classifier(x)
        
        print(score.shape)
        
        return score