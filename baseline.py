import torch
import torch.nn as nn

class CNNBaseline(nn.Module):
    def __init__(self, input_size=(3, 224, 224)) -> None:
        super().__init__()
        self.encoders_in = [3, 32, 64, 128, 256]
        self.encoders_out = [32, 64, 128, 256, 512]
        
        self.decoders_in = [512, 256, 128, 64]
        self.decoders_out = [256, 128, 64, 32]
        
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
        
        # Dummy input to calculate the size of the flattened features dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            self.feature_size = self._get_conv_output(dummy_input).nelement()
        
        self.classifier = nn.Linear(self.feature_size, 5)

    def _get_conv_output(self, x):
        for layer in self.conv:
            x = layer(x)
        for layer in self.deconv:
            x = layer(x)
        return x

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        
        for layer in self.deconv:
            x = layer(x)
            
        #print(f'Before flatten: {x.shape}')
        
        x = torch.flatten(x, 1)
        
        #print(f'After flatten: {x.shape}')
        #print(self.classifier)
        score = self.classifier(x)
        return score
