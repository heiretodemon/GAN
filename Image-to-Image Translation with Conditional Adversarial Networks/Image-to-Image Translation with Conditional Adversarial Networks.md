[TOC]

# Image-to-Image Translation with Conditional Adversarial Networks

[paper link](http://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)

![](https://github.com/heiretodemon/GAN/blob/master/Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks/1.png/1.png)

## Abstract 

```latex
We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping.
```

## Background 

&emsp; Many problems in image processing, computer graphics, and computer vision can be posed as "translating" an input image into a corresponding output image. The usually approach is to use CNNs to learn to minimize a loss function but a lot of manual effort still goes into designing effective losses. And the naïve approach which ask the CNN to minimize Euclidean distance will tend to produce blurry results. However, what we really want is to output sharp and realistic images.



## Method

&emsp; Conditional GANs learn a mapping from observed image $x$ and random noise vector $z$, to $y, G :\{x, z\} \rightarrow y$. The objective of a conditional GAN can be expressed as :

​                               $\begin{aligned} \mathcal{L}_{c G A N}(G, D)=& \mathbb{E}_{x, y}[\log D(x, y)]+\\ & \mathbb{E}_{x, z}[\log (1-D(x, G(x, z))]\end{aligned}​$

and $G^*=\arg \min _{G} \max _{D} \mathcal{L}_{c G A N}(G, D)​$

&emsp; To encourage less blurring, we use $L1$ distance rather than $L2$ :

​                                $\mathcal{L}_{L 1}(G)=\mathbb{E}_{x, y, z}\left[\|y-G(x, z)\|_{1}\right]$

&emsp; So our final objective is
$G^{*}=\arg \min _{G} \max _{D} \mathcal{L}_{c G A N}(G, D)+\lambda \mathcal{L}_{L 1}(G)​$

## Generator

&emsp; Here we use Encoder-decoder network and skip connections between each layer $i$ and layer $n-i$, which creates a "U-Net" results in much higher quality results.

## Markovian discriminator(PatchGAN)

&emsp; In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. Therefore, we design a discriminator architecture- which we term a PatchGAN($N \times N$ patch)- that only penalizes structure at the scale of patches. In this paper, we demonstrate that N can be much smaller than the full size of the image and still produce high quality results, because of its fewer parameters and runs faster.

 
