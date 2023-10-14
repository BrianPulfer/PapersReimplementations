# Import of libraries
import random
from argparse import ArgumentParser

import einops
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from tqdm.auto import tqdm

# Import of custom models
from src.ddpm.models import MyDDPM, MyUNet

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"


def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break


def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(
                    imgs.to(device),
                    [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))],
                ),
                f"DDPM Noisy images {int(percent * 100)}%",
            )
        break


def generate_new_images(
    ddpm,
    n_samples=16,
    device=None,
    frames_per_gif=100,
    gif_name="sampling.gif",
    c=1,
    h=28,
    w=28,
):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(
                    normalized,
                    "(b1 b2) c h w -> (b1 h) (b2 w) c",
                    b1=int(n_samples**0.5),
                )
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Showing the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x


def training_loop(
    ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"
):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(
            tqdm(
                loader,
                leave=False,
                desc=f"Epoch {epoch + 1}/{n_epochs}",
                colour="#005500",
            )
        ):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(
                generate_new_images(ddpm, device=device),
                f"Images generated at epoch {epoch + 1}",
            )

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


def main():
    # Program arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--no_train", action="store_true", help="Whether to train a new model or not"
    )
    parser.add_argument(
        "--fashion",
        action="store_true",
        help="Uses MNIST if true, Fashion MNIST otherwise",
    )
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = vars(parser.parse_args())
    print(args)

    # Model store path
    store_path = "ddpm_fashion.pt" if args["fashion"] else "ddpm_mnist.pt"

    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    ds_fn = MNIST if not args["fashion"] else FashionMNIST
    dataset = ds_fn("./../datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, args["bs"], shuffle=True)

    # Getting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device}\t"
        + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
    )

    # Defining model
    n_steps, min_beta, max_beta = 1000, 10**-4, 0.02  # Originally used by the authors
    ddpm = MyDDPM(
        MyUNet(n_steps),
        n_steps=n_steps,
        min_beta=min_beta,
        max_beta=max_beta,
        device=device,
    )

    # Optionally, load a pre-trained model that will be further trained
    # ddpm.load_state_dict(torch.load(store_path, map_location=device))

    # Optionally, show a batch of regular images
    # show_first_batch(loader)

    # Optionally, show the diffusion (forward) process
    # show_forward(ddpm, loader, device)

    # Optionally, show the denoising (backward) process
    # generated = generate_new_images(ddpm, gif_name="before_training.gif")
    # show_images(generated, "Images generated before training")

    # Training
    if not args["no_train"]:
        n_epochs, lr = args["epochs"], args["lr"]
        training_loop(
            ddpm,
            loader,
            n_epochs,
            optim=Adam(ddpm.parameters(), lr),
            device=device,
            store_path=store_path,
        )

    # Loading the trained model
    best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("Model loaded: Generating new images")

    # Showing generated images
    generated = generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name="fashion.gif" if args["fashion"] else "mnist.gif",
    )
    show_images(generated, "Final result")


if __name__ == "__main__":
    main()
