[TOC]

#  Visual Object Networks: Image Generation with Disentangled 3D Representation

## original paper

[paper link](<http://papers.nips.cc/paper/7297-visual-object-networks-image-generation-with-disentangled-3d-representations>)

## Introduction
&emsp; In this paper, we present an end-to-end generative model that jointly synthesizes 3D shapes and 2D images via a disentangled object representation. Speciﬁcally, we decompose our image generation model into three conditionally independent factors: shape, viewpoint, and texture, borrowing ideas from classic graphics rendering engines. This model allows a user to change the viewpoint easily, as well as to edit the object’s shape or texture in dependently.

## Method

![](https://github.com/heiretodemon/GAN/blob/master/Visual%20Object%20Networks/1.png)

### Learning 3D shape Priors

- We adopt the [3D-GAN](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf) to model the 3D shape prior  and generate realistic shapes.

- Both $G$ and $D$ contain fully volumetric convolutional  and deconvolutional layers.

- To solve mode collapse and improve the quality and diversity of the results, we use the Wasserstein distance of WAGN-GP, so the loss function is:

$$
\mathcal{L}_{\text { shape }}=\mathbb{E}_{\mathbf{v}}\left[D_{\text { shape }}(\mathbf{v})\right]-\mathbb{E}_{\mathbf{z}_{\text { stape }}}\left[D_{\text { shape }}\left(G_{\text { shape }}\left(\mathbf{z}_{\text { shape }}\right)\right]\right.
$$

- To enforce the Lipschitz constraint in Wasserstein GANs , we add a gradientpenalty loss $$\lambda_{\mathrm{GP}} \mathbb{E}_{\tilde{\mathbf{v}}}\left[\left(\nabla_{\tilde{\mathbf{v}}} D_{\text { shape }}(\tilde{\mathbf{v}})-1\right)^{2}\right]$$ to $\mathcal{L}_{\text { shape }}$, where $\tilde{\mathbf{V}}$ is a randomly sampled point along the straight line between a real shape and a generated shape, and $\lambda_{\mathrm{GP}}$ controls the capacity of $D_{\text { shape }}$. 

### Generating 2.5D Sketches

&emsp; Inspired by recent work on 3D reconstruction, we use 2.5D sketches to bridge the gap between 3D and 2D. This intermediate representation provides 3 advantages:

- Straightforward;
- There are some existing methods have achieved successes even without paired data like CycleGAN;
- Generating images at a higher resolution.

Input: camera parameters and 3D voxels, the value of each voxel stores the probability of it being present.

Key point: Generating a collection of rays, each originating from the camera's center and going through a pixel's center in the image plane, and calculate whether a given ray would hit the voxels.

Process: 

- Sample a collection of points at spaced depth along each ray;

- Calculating the probability of hitting the input voxels using a differentiable trilinear interpolation of the input voxels;

- Calculating the expectation of visibility and depth along each ray:

  Expectation and depth:

  $$
  \sum_{j=1}^{N} \prod_{k=1}^{j-1}\left(1-R_{k}\right) R_{j}​\\
  \sum_{j=1}^{N} d_{j} \prod_{k=1}^{j-1}\left(1-R_{k}\right) R_{j}
  $$

#### Viewpoint estimation

&emsp; Code $Z_{view}$ encodes camera elevation and azimuth.  

- We sample $Z_{view}$ from an empirical distribution $p_{data}(Z_{view})$ of the camera poses;
- Rendering the silhouettes of several candidate 3D model;
- Comparing its silhouette to the rendered 2D views and choose the pose with the largest IoU value.

### Learning 2D Texture Priors

&emsp; To synthesize realistic 2D images  given projected 2.5D sketches that encode both the viewpoint and the object shape. so the 2D images $\mathbf{x}=G_{\text { texture }}\left(\mathbf{v}_{2.5 \mathrm{D}}, \mathbf{Z}_{\text { texture }}\right)​$.  This network needs to model both object texture and environment illumination, and this mapping problem can be cast as an unpaired image-to-image translation problem, so CycleGAN is our baseline.

&emsp; So the loss functions are:

$$
The\space adversarial\space loss\space on\space image: \mathcal{L}_{\text { image }}^{\text { GAN }}=\mathbb{E}_{\mathbf{x}}\left[\log D_{\text { image }}(\mathbf{x})\right]+\mathbb{E}_{\left(\mathbf{v}_{25 \mathbf{D}}, \mathbf{z}_{\text { texure }}\right)}\left[\log \left(1-D_{\text { image }}\left(G_{\text { texture }}\left(\mathbf{v}_{2.5 \mathrm{D}}, \mathbf{Z}_{\text { texture }}\right)\right)\right)\right.\\
The\space same\space loss\space on\space sketches: \mathcal{L}_{2.5 \mathrm{D}}^{\mathrm{GAN}}=\mathbb{E}_{\mathrm{v}_{25 \mathrm{D}}}\left[\log D_{2.5 \mathrm{D}}\left(\mathbf{v}_{2.5 \mathrm{D}}\right)\right]+\mathbb{E}_{\mathrm{x}}\left[\log \left(1-D_{2.5 \mathrm{D}}\left(E_{2.5 \mathrm{D}}(\mathrm{x})\right)\right]\right.\\

The\space cycle-loss\space loss: \mathcal{L}_{2.5 \mathrm{D}}^{\mathrm{cyc}}=\lambda_{2.5 \mathrm{D}}^{\mathrm{cyc}} \mathbb{E}_{\left(\mathbf{v}_{2.5 \mathrm{D}}, \mathbf{z}_{\mathrm{texture}}\right)}\left[\left\|E_{2.5 \mathrm{D}}\left(G_{\mathrm{texture}}\left(\mathbf{v}_{2.5 \mathrm{D}}, \mathbf{Z}_{\mathrm{texture}}\right)\right)-\mathbf{v}_{2.5 \mathrm{D}}\right\|_{1}\right]\\
and\space \mathcal{L}_{\text { image }}^{\mathrm{cyc}}=\lambda_{\mathrm{image}}^{\mathrm{cyc}} \mathbb{E}_{\mathbf{x}}\left[ \| G_{\text { texture }}\left(E_{2.5 \mathrm{D}}(\mathbf{x}), E_{\text { texture }}(\mathbf{x})\right)-\mathbf{x}\left\|_{1}\right]\right.
$$

#### One-to-many mappings

&emsp;We introduce a latent space cycle-consistency loss to encourage $G_{texture}$ to use the texture code $Z_{texture}$: 

$$
\mathcal{L}_{\text { texture }}^{\mathrm{cyc}}=\lambda_{\text { texture }}^{\mathrm{cyc}} \mathbb{E}_{\left(\mathbf{v}_{2.5 \mathrm{D}}, \mathbf{z}_{\text { texture }}\right)}\left[ \| E_{\text { texture }}\left(G_{\text { texture }}\left(\mathbf{v}_{2.5 \mathrm{D}}, \mathbf{Z}_{\text { texture }}\right)\right)-\mathbf{Z} \text { texture }\left\|_{1}\right]\right.
$$
&emsp;Finally, to allow sampling at test time, we add a Kullback–Leibler (KL) loss on the $z$ space to force $E_{texture(x)}​$ to be close to a Gaussian distribution: 

$$
\mathcal{L}_{\mathrm{KL}}=\lambda_{\mathrm{KL}} \mathbb{E}_{\mathbf{x}}\left[\mathcal{D}_{\mathrm{KL}}\left(E_{\text { texture }}(\mathbf{x}) \| \mathcal{N}(0, I)\right)\right]
$$
So the final texture loss is :

$$
\mathcal{L}_{\text { texture }}=\underbrace{\mathcal{L}_{\text { image }}^{\text { GAN }}+\mathcal{L}_{2.5 \mathrm{D}}^{\text { cyc }}}+\underbrace{\mathcal{L}_{2.5 \mathrm{D}}^{\text { cyc }}+\mathcal{L}_{\text { texture }}^{\text { cyc }}+\mathcal{L}_{\text { texture }}}_{\text { Cycle-consistency losses }}+\underbrace{\mathcal{L}_{\mathrm{KL}}}_{\text { KL loss }}
$$

### Full model

&emsp; Our full objective is:

$$
argmin_{(G_{shape}, G_{texture}, E_{2.5D},E_{texture})} argmax_{(D_{shape}, D_{texture}, D_{2.5D})} \lambda \mathcal{L}_{shape} + \mathcal{L}_{texture}
$$

## Experiment

![](https://github.com/heiretodemon/GAN/blob/master/Visual%20Object%20Networks/2.png)

![](https://github.com/heiretodemon/GAN/blob/master/Visual%20Object%20Networks/3.png)

![](https://github.com/heiretodemon/GAN/blob/master/Visual%20Object%20Networks/4.png)
