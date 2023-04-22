import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


# Baseline model
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)  # Pretrained on resnet, 50 layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1)
        self.fc1 = nn.Linear(512 * 4, 256)
        self.fc2 = nn.Linear(512 * 4, 256)

        del self.resnet.fc
        del self.resnet.avgpool

        self.lin_1 = torch.nn.Sequential(
            torch.nn.Linear(256,100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU6(),
        )
        self.lin_2 = torch.nn.Sequential(
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)
        rep = torch.flatten(x,1)
        x = self.fc1(rep)

        x = self.lin_1(x)
        x = self.lin_2(x)
        return x


# Encoder for $Q_{Z|X}
class ResnetEncoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.resnet = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1)
        self.fc1 = nn.Linear(512 * 4, latent_dim)
        self.fc2 = nn.Linear(512 * 4, latent_dim)
        del self.resnet.fc
        del self.resnet.avgpool

    # Reparameterization trick using variance for when alpha > 1
    def reparameterize_highalpha(self,mu,var):
        std = var ** 0.5
        # random normal distribution
        eps = torch.randn_like(std, device=self.device)
        return mu + std * eps

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)

        rep = torch.flatten(x,1)
        mu = self.fc1(rep)

        var = self.fc2(rep)

        # sigmoid added to keep variance between 0 and 1 to allow for alpha > 1
        # for renyi divergence
        # var = torch.sigmoid(var)

        # for renyi cross entropy we want the variance to be > 1
        var = torch.clamp(var, min=1)


        z = self.reparameterize_highalpha(mu,var)
        return z, mu, var


# Decoder for Q_{Y|Z}
class Decoder(nn.Module):
    def __init__(self,latent_dim,output_dim):
        super().__init__()

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, output_dim),
        )

    def forward(self,z):
        yhat = self.lin1(z)
        return yhat


# Decoder for Q_{Y|S,Z}
class FairDecoder(nn.Module):
    def __init__(self,latent_dim,output_dim):
        super().__init__()

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+1, output_dim),
        )

    def forward(self, z, s):
        s = s.view(-1, 1)
        yhat_fair = self.lin1(torch.cat((z, s), 1))
        return yhat_fair


# Decoder for Q_{X|P,Z}
class PrivateDecoder(nn.Module):
    def __init__(self,latent_dim,img_dim=256):
        super().__init__()

        fc_hidden1 = 1024
        fc_hidden2 = 768
        self.img_dim = img_dim

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, latent_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim + 1, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()
        )

    def forward(self, z, p):
        p = p.view(-1, 1)
        cat = torch.cat((z, p), 1)

        x = self.relu(self.fc_bn4(self.fc4(cat)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(self.img_dim, self.img_dim), mode='bilinear')
        return x

# Main RFPIB model. Subsumes IB, CFB, and RFIB
class RFPIB(nn.Module):
    def __init__(self,latent_dim,img_dim=256,output_dim=1):
        super().__init__()

        self.encoder = ResnetEncoder(latent_dim)
        self.decoder = Decoder(latent_dim,output_dim)
        self.fair_decoder = FairDecoder(latent_dim,output_dim)
        self.private_decoder = PrivateDecoder(latent_dim,img_dim)

    def forward(self, x, s, p):
        z, mu, logvar = self.encoder(x)
        yhat = self.decoder(z)
        yhat_fair = self.fair_decoder(z, s)
        reconstruction = self.private_decoder(z, p)

        return yhat, yhat_fair, mu, logvar, reconstruction

    def getz(self,x):
        return self.encoder(x)

    def get_reconstruction(self,z,p):
        return self.private_decoder(z,p)


