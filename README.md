# Description
Personal re-implementations of known Machine Learning architectures, layers, algorithms and more.
Re-implementations might be semplified and approximate. The goal is getting the core concepts down 🙂.

## Package transformer
Implementation of the "_Attention is all you need_" [paper](https://arxiv.org/abs/1706.03762) 

## Package vit
Implementation of the "_An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_" [paper](https://openreview.net/forum?id=YicbFdNTTy).
The MNIST dataset is used as a toy example for classification task.

## Package ddpm
Implementation of the "_Denoising Diffusion Probabilistic Models_" [paper](https://arxiv.org/abs/2006.11239).
I use MNIST and FashionMNIST dataset as toy examples. The model used is a custom U-Net like architecture with the use of positional embeddings.
Pre-trained models for both datasets (20 epochs only) are provided in the package when using [Git Large File System](https://git-lfs.github.com/).

<img src="./ddpm/both.gif" />