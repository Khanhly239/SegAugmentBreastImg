import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        # Concatenate input and target images
        x = torch.cat([img_A, img_B], 1)
        return self.model(x)

def train_step(input_image, target, generator, discriminator, g_optimizer, d_optimizer, criterion):
    # Train Discriminator
    discriminator.zero_grad()
    real_output = discriminator(input_image, target)
    fake_target = generator(input_image)
    fake_output = discriminator(input_image, fake_target.detach())
    d_loss_real = criterion(real_output, torch.ones_like(real_output))
    d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
    d_loss = (d_loss_real + d_loss_fake) * 0.5
    d_loss.backward()
    d_optimizer.step()

    # Train Generator
    generator.zero_grad()
    fake_output = discriminator(input_image, fake_target)
    g_loss_gan = criterion(fake_output, torch.ones_like(fake_output))
    g_loss_l1 = nn.L1Loss()(fake_target, target)
    g_loss = g_loss_gan + 100 * g_loss_l1
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item(), d_loss.item()

def fit(train_loader, epochs, generator, discriminator, g_optimizer, d_optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(epochs):
        for n, (input_image, target) in enumerate(train_loader):
            input_image, target = input_image.to(device), target.to(device)
            g_loss, d_loss = train_step(input_image, target, generator, discriminator, g_optimizer, d_optimizer, criterion)
            if n % 10 == 0:
                print('.', end='')
        print()
        print('Epoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch, g_loss, d_loss))
        # if (epoch + 1) % 20 == 0 or epoch == 0:
        #   torch.save(generator.state_dict(), f'generator{epoch+1}.pth')
        # Save sample generated images
        with torch.no_grad():
            fake_target = generator(input_image)
            save_image(fake_target, 'generated_images/epoch_{}.png'.format(epoch))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate models
generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd

class BUS_loader(Dataset):
    def __init__(self, iw=512, ih=512):
        super().__init__()
        self.iw = iw
        self.ih = ih
        self.csv_file = "/content/BUSBRA/bus_data.csv"
        self.img_normalization = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mask_normalization = transforms.Compose([transforms.ToTensor()])
        self.__get_path__()
        
    def __len__(self):
        return len(self.img)

    def __gen_train_test(self):
        df = pd.read_csv(self.csv_file)
        
        train_img = df['ID'].tolist()
        self.train_img = [f'{item}.png' for item in train_img]
        self.train_target = [f'{item}.png' for item in train_img]
    
    def __get_path__(self):
        self.__gen_train_test()
        self.img = self.train_img
        self.target = self.train_target
    
    def __getitem__(self, index):
        target_path = '/content/BUSBRA/Images/' + self.target[index]
        target = cv2.imread(target_path,cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (self.iw, self.ih))
        img_path = '/content/BUSBRA/Masks/mask_' + self.img[index].split("_")[1]
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(self.iw,self.ih),interpolation=cv2.INTER_NEAREST)  
        img = self.mask_normalization(img)
        target = self.img_normalization(target)
        
        return img, target

train_dataset = BUS_loader()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
# Assume train_loader is your DataLoader for the dataset
fit(train_loader, 200, generator, discriminator, g_optimizer, d_optimizer, criterion)

# Save the generator model
torch.save(generator.state_dict(), 'bus_generator.pth')

# # Load the generator model for inference
bus_generator = GeneratorUNet()
bus_generator.load_state_dict(torch.load('bus_generator.pth'))
bus_generator.to(device)
bus_generator.eval()

# # Generate images for visualization
with torch.no_grad():
    for input_image, target in train_loader:
        input_image = input_image.to(device)
        fake_image = bus_generator(input_image)
        save_image(fake_image.cpu(), 'generated_images/sample.png', normalize=True)
        break  
