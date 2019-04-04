[TOC]

#  Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling

## original paper

[paper link](<http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf>)

## Introduction

&emsp; We propose a novel framework, namely 3D Generative Adversarial Network(3D-GAN), which generates 3D-objects form a probabilistic space by leveraging recent advances in *volumetric convolutional networks and generative adversarial nets*. 

## Advantages

- Use of an adversarial criterion, enables the generator to capture object structure implicitly and to synthesize high-quality 3D objects;

- Our model becomes possible to sample novel 3D objects from a probabilistic latent space such as a Gaussian or uniform distribution;

- The discriminator in the generative-adversarial approach carries informative features for 3D object recognition;

- Generator establishes a mapping form a low-dimensional probabilistic space to the space of 3D objects, so tha t we can sample objects without a reference image or CAD models, and explore the 3D object manifold;

  

## Models

###  3D-GAN

&emsp;So the loss function is 

$$
L_{3 \mathrm{D}-\mathrm{GAN}}=\log D(x)+\log (1-D(G(z)))
$$
where $x​$ is a real object in a $64 \times 64 \times 64​$ space, and $z​$ is a randomly sampled noise vector from a distribution $p(z)​$. In this work, each dimension of $z​$ is an i.i.d. uniform distribution over [0,1].

#### Network structure 

&emsp; The generator consists of five volumetric fully convolutional layers of kernel sizes $4 \times 4\times 4​$ and strides $2​$, whcich bat ch normalization and ReLU layers added in between and a Sigmoid layer at the end. The discirminator basically mirrors the generator, except that it uses Leaky ReLU instead of ReLU layers. There are no pooling or linear layers in our network.

### 3D-VAE-GAN

&emsp;We introduce 3D-VAE-GAN as an extension to 3D-GAN. We addd an additional image encoder $E$, which takes 2D image $x$ as input and outputs the latent representation vector $z$. That means it consists of 3 components: an image encoder $E$,  a decoder (the generator $G$ in 3D-GAN), and a discriminator $D$.

&emsp; Our loss function consists of 3 parts: an object reconstruction loss $L_{recon}$ a cross entropy loss $L_{3D-GAN}$ for 3D-GAN, and a KL divergence loss $L_{KL}$ to restrict the distribution of the output of the encoder.
$$
L=L_{3 \mathrm{D}-\mathrm{CAN}}+\alpha_{1} L_{\mathrm{KL}}+\alpha_{2} L_{\mathrm{reon}}
$$
where $\alpha_1$ and $\alpha_2$ are weights of the KL divergence loss and reconstruction loss. We have 
$$
\begin{aligned} L_{3 \mathrm{D}-\mathrm{GAN}} &=\log D(x)+\log (1-D(G(z))) \\ L_{\mathrm{KL}} &=D_{\mathrm{KL}}(q(z | y) \| p(z)) \\ L_{\mathrm{recon}} &=\|G(E(y))-x\|_{2} \end{aligned}
$$
where $x$ is a 3D shape from the training set, $y$ is its corresponding 2D image, and $q(z|y)$ is the variational distribution of the latent representation $z​$.

# Evaluation

![1554370960567](C:\Users\Wolf\AppData\Roaming\Typora\typora-user-images\1554370960567.png)