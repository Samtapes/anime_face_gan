import os

import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
import torch.nn as nn

import torch.nn.functional as F


## IMPORTING DATA
data_dir = 'anime_gan/data'

image_size = 64
batch_size = 128
stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

dataset = ImageFolder(data_dir, transform=Compose([Resize(image_size), CenterCrop(image_size), ToTensor(), Normalize(*stats)]))

data_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)



## HELPER FUNCTIONS
def denorm(img_tensors):
  return img_tensors * stats[1][0] + stats[0][0]


def get_default_device():
  """Pick GPU if available, else CPU"""
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')
    
def to_device(data, device):
  """Move tensor(s) to chosen device"""
  if isinstance(data, (list,tuple)):
      return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)

class DeviceDataLoader():
  """Wrap a dataloader to move data to a device"""
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device
      
  def __iter__(self):
    """Yield a batch of data after moving it to device"""
    for b in self.dl: 
        yield to_device(b, self.device)

  def __len__(self):
    """Number of batches"""
    return len(self.dl)


device = get_default_device()

data_loader = DeviceDataLoader(data_loader, device)




## MODELS

D = nn.Sequential(
  # in: 3 x 64 x 64

  nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(64),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 64 x 32 x 32

  nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(128),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 128 x 16 x 16

  nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(256),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 256 x 8 x 8

  nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(512),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 512 x 4 x 4

  nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
  # out: 1 x 1 x 1

  nn.Flatten(),
  nn.Sigmoid()
)

D = to_device(D, device)



latent_size = 128

G = nn.Sequential(
  # in: latent_size x 1 x 1

  nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
  nn.BatchNorm2d(512),
  nn.ReLU(True),
  # out 512 x 4 x 4

  nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  # out 256 x 8 x 8

  nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  # out 128 x 16 x 16

  nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  # out 64 x 32 x 32

  nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
  nn.Tanh()
  # out 3 x 64 x 64
)

G = to_device(G, device)



## TRAINING THE MODELS
criterion = F.binary_cross_entropy


def train_discriminator(images, d_optimizer):
  d_optimizer.zero_grad()

  # Labels to calculate loss
  real_labels = torch.ones(images.size(0), 1).to(device)
  fake_labels = torch.zeros(batch_size, 1).to(device)

  # Training with real images
  out = D(images)
  d_loss = criterion(out, real_labels)
  real_score = d_loss

  # Training with fake images
  latent = torch.randn(batch_size, latent_size, 1, 1).to(device)
  fake_images = G(latent)
  g_loss = criterion(D(fake_images), fake_labels)
  fake_score = g_loss

  # Optimizing
  loss = d_loss + g_loss
  loss.backward()
  d_optimizer.step()

  return loss, real_score, fake_score



def train_generator(g_optimizer):
  g_optimizer.zero_grad()

  # Generating Fake Images
  latent = torch.randn(batch_size, latent_size, 1, 1).to(device)
  fake_images = G(latent)

  # Try to fool the discriminator
  labels = torch.ones(batch_size, 1).to(device)
  g_loss = criterion(D(fake_images), labels)

  # Optimizing
  g_loss.backward()
  g_optimizer.step()

  return g_loss, fake_images




## SAVING IMAGES
import os

sample_dir = 'anime_gan/samples2'
if not os.path.exists(sample_dir):
  os.makedirs(sample_dir)


from torchvision.utils import save_image


#  Save some real images
for images, _ in data_loader:
  images = images.reshape(images.size(0), 3, 64, 64)
  print(images.size())
  save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'), nrow=10)
  break



sample_vectors = torch.randn(batch_size, latent_size, 1, 1).to(device)

def save_fake_images(index):
  fake_images = G(sample_vectors)
  fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
  fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
  print('Saving', fake_fname)
  save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)

# Before training
save_fake_images(0)




## TRAINING THE MODEL
total_step = len(data_loader)

def fit(num_epochs, lr):
  torch.cuda.empty_cache()

  d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
  g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

  for epoch in range(num_epochs):
    print("NEW EPOCH")
    for i, (images, _) in enumerate(data_loader):

      # Train the discriminator and generator
      d_loss, real_score, fake_score = train_discriminator(images, d_optimizer)
      g_loss, fake_images = train_generator(g_optimizer)

      # Inspect the losses
      if (i+1) % 200 == 0:
        print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))

    # Sample and save images
    save_fake_images(epoch+1)


fit(25, lr=0.0002)