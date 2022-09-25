"""
Personal reimplementation of 
    Density estimation using Real NVP
(https://arxiv.org/abs/1605.08803)

Usefule links:
     - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
     - https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/affine/coupling.py
"""

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor, Compose, Lambda

from borrow import Dequantization as BorrowedDequant

# Seeding
np.random.seed(0)
torch.random.manual_seed(0)


def test_reversability(model, x):
    """Tests that x ≈ model.backward(model(x)) and shows images"""
    with torch.no_grad():
        # Running input forward and backward
        z = model.forward(x)[0]
        x_tilda = model.backward(z)[0]
        
        # Printing MSE
        mse = ((x_tilda - x)**2).mean()
        print(f"MSE between input and reconstruction: {mse}")
        
        # Comparing images visually
        plt.imshow(x[0][0].cpu().numpy(), cmap="gray")
        plt.title("Original image")
        plt.show()
        
        plt.imshow(z[0][0].cpu().numpy(), cmap="gray")
        plt.title("After forward pass")
        plt.show()
        
        plt.imshow(x_tilda[0][0].cpu().numpy(), cmap="gray")
        plt.title("Reconstructed image")
        plt.show()

def new_convnet(k_size = 3):
    """Returns a simple convnet"""
    assert k_size % 2 == 1, "Kernel size must be odd"
    
    return nn.Sequential(
        nn.LayerNorm((1, 28, 28)),
        nn.Conv2d(1, 2, k_size, 1, k_size // 2),
        nn.GELU(),
        nn.Conv2d(2, 2, k_size, 1, k_size // 2),
        nn.GELU(),
        nn.Conv2d(2, 2, k_size, 1, k_size // 2),
        nn.GELU(),
        nn.Conv2d(2, 2, k_size, 1, k_size // 2)
    )

class MyDequantization(nn.Module):
    """Dequantizes the image. Dequantization is the first step for flows, as it allows to not load datapoints
    with high likelihoods and put volume on other input data as well."""
    def __init__(self):
        super(MyDequantization, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        
    def inv_sigmoid(self, x):
        return - torch.log(1 / x - 1)
        
    def forward(self, x):
        out = x + torch.rand_like(x).detach() / 256 
        out = self.sigmoid(out)
        return out, torch.log(1-torch.exp(-x)).sum(dim=[1, 2, 3])
    
    def backward(self, x):
        out = self.inv_sigmoid(x)
        return out, torch.log(x**-2 / (x**-1 -1)).sum(dim=[1, 2, 3])

class AffineCoupling(nn.Module):
    """Affine Coupling layer. Only modifies half of the input by running the other half through some non-linear function."""
    def __init__(self, m : nn.Module, modify_x2 = True):
        super(AffineCoupling, self).__init__()
        self.m = m
        self.modify_x2 = modify_x2
        self.s_fac = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Splitting input in two halves
        splitted = x.chunk(2, 1) if self.modify_x2 else x.chunk(2, 1)[::-1]
        x1, x2 = splitted
        
        # Computing scale and shift for x2
        scale, shift = self.m(x1).chunk(2, 1) # Non linear network
        scale = torch.tanh(scale / self.s_fac) * self.s_fac  # Stabilizes training
        
        # Computing output
        y1 = x1
        y2 = torch.exp(scale) * x2 + shift
        out = torch.cat((y1, y2), 1)
        
        # Computing log of the determinant of the Jacobian
        log_det_j = torch.sum(scale, dim=[1, 2, 3])
        
        return out, log_det_j
    
    def backward(self, y):
        # Splitting input
        y1, y2 = y.chunk(2, 1)
        
        # Computing scale and shift
        scale, shift = self.m(y1).chunk(2, 1)
        scale = torch.tanh(scale / self.s_fac) * self.s_fac
        
        # Computing inverse transformation
        x1 = y1
        x2 = (y2 - shift) / torch.exp(scale)
        out = torch.cat((x1, x2), 1) if self.modify_x2 else torch.cat((x2, x1), 1)
        
        # Computing log of the determinant of the Jacobian (for backward tranformation)
        log_det_j =  -torch.sum(scale, dim=[1, 2, 3])
        
        return out, log_det_j
    
class Flow(nn.Module):
    """General Flow model. Uses invertible layers to map distributions."""
    def __init__(self, layers):
        super(Flow, self).__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        # Computing forward pass (images --> gaussian noise)
        out, log_det_j = x, 0
        for layer in self.layers:
            out, log_det_j_layer = layer(out)
            log_det_j += log_det_j_layer
            
        return out, log_det_j
    
    def backward(self, y):
        # Sampling with backward pass (gaussian noise --> images)
        out, log_det_j = y, 0
        for layer in self.layers[::-1]:
            out, log_det_j_layer = layer.backward(out)
            log_det_j += log_det_j_layer
            
        return out, log_det_j
    
def training_loop(model, epochs, lr, wd, loader, device):
    """Trains the model"""
    
    model.train()
    best_loss = float("inf")
    optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
    to_bpd = np.log2(np.exp(1)) / (28*28*1) # Constant that normalizes w.r.t. input shape
    
    for epoch in tqdm(range(epochs), desc="Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for batch in tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{epochs}", colour="#005500"):
            # Getting a batch of images and applying dequantization
            x = batch[0].to(device)
            
            # Running images forward and getting log likelihood (log_px)
            z, log_det_j = model(x)
            log_pz = np.log(1/np.sqrt(2*np.pi)) -(z**2).sum(dim=[1,2,3]) # Because we are mapping to a normal N(0, 1)
            log_px = log_pz + log_det_j
            
            # Getting the loss to be optimized (scaling with bits per dimension)
            loss = (-(log_px * to_bpd)).mean()
            
            # Optimization step
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # Logging variable
            epoch_loss += loss.item() / len(loader)
        
        # Logging epoch result and storing best model
        log_str = f"Epoch {epoch+1}/{epochs} loss: {epoch_loss:.3f}"
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            log_str += " --> Storing model"
            
            torch.save(model.state_dict(), "./nf_model.pt")
        print(log_str)

def main():
    # Program arguments
    N_EPOCHS = 20
    LR = 0.001
    WD = 0.9
    
    # Loading data (images are in range [0, 1])
    # TODO: Set images in range [0, 255]
    transform = Compose([ToTensor(), Lambda(lambda x: 255 * x.repeat_interleave(2, 0))])
    dataset = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Creating the model
    model = Flow([
        BorrowedDequant(),
        AffineCoupling(new_convnet(7), modify_x2=True),
        AffineCoupling(new_convnet(7), modify_x2=False),
        AffineCoupling(new_convnet(5), modify_x2=True),
        AffineCoupling(new_convnet(5), modify_x2=False),
        # AffineCoupling(new_convnet(3), modify_x2=True),
        # AffineCoupling(new_convnet(3), modify_x2=False),
    ]).to(device)
    
    # Testing reversability with first image in the dataset
    test_reversability(model, dataset[0][0].unsqueeze(0).to(device))
    
    # Training loop
    training_loop(model, N_EPOCHS, LR, WD, loader, device)
    
    # Testing the trained model
    model.load_state_dict(torch.load("./nf_model.pt"))
    model.eval()
    with torch.no_grad():
        # Mapping the normally distributed noise to new images
        noise = torch.randn(64, 2, 28, 28).to(device)
        images = model.backward(noise)[0]  
        images = images[:, 0, :, :].reshape(64, 1, 28, 28)  # Removing duplicate channel
    
    save_image(images.float(), "Generated digits.png")
    Image.open("Generated digits.png").show()
    
    # Showing new latent mapping of first image in the dataset
    test_reversability(model, dataset[0][0].unsqueeze(0).to(device))

if __name__ == "__main__":
    main()
