import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            # 8*16*16
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*8*8
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32*4*4
        ]
        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(32*4*4, 1)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1) # Flatten tensor
        x = self.fc(x)
        return x 