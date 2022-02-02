import torch
import torch.nn as nn

import numpy as np
from PIL import Image


def show_tensor_image(tensor):
    Image.fromarray((tensor.numpy() * 255).astype(np.uint8)).show()


class AdaIN(nn.Module):
    def __init__(self, mu, sigma):
        super(AdaIN, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        norm_x = (x - torch.mean(x, axis=0)) / torch.std(x, axis=0)
        return self.sigma * norm_x + self.mu


def main():
    # Loading content and style (resized) images
    content = torch.Tensor(np.array(Image.open("content.jpeg")) / 255.0)
    h, w = content.shape[0], content.shape[1]
    style = torch.Tensor(np.array(Image.open("style.jpeg").resize((w, h))) / 255.0)

    # Creating AdaIN layer
    adain = AdaIN(
        mu=torch.mean(style, axis=0),
        sigma=torch.mean(style, axis=0)
    )

    # Applying AdaIN layer
    mix = adain(content)

    # Showing result of applying AdaIN
    show_tensor_image(content)
    show_tensor_image(style)
    show_tensor_image(mix)


if __name__ == '__main__':
    main()
