# Description
Personal re-implementations of known Machine Learning architectures, layers, algorithms and more.
Re-implementations might be simplified and approximate. The goal is learning the core concepts 🙂.

## Package transformer
Implementation of the "_Attention is all you need_" [paper](https://arxiv.org/abs/1706.03762). I also made a small [Blog](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c). 

## Package vit
Implementation of the "_An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_" [paper](https://openreview.net/forum?id=YicbFdNTTy).
The MNIST dataset is used as a toy example for classification task.

## Package ddpm
Implementation of the "_Denoising Diffusion Probabilistic Models_" [paper](https://arxiv.org/abs/2006.11239).
I use MNIST and FashionMNIST dataset as toy examples. The model used is a custom U-Net like architecture with the use of positional embeddings.
Pre-trained models for both datasets (20 epochs only) are provided in the package when using [Git Large File System](https://git-lfs.github.com/).
Check out the [Blog](https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1) for a step-by-step explanation.

<img src="./ddpm/both.gif" />