[TOC]

# Conditional Generative Adversarial Network for Structured Domain Adaption

[paper link](http://openaccess.thecvf.com/content_cvpr_2018/html/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.html)

## Abstract 

&emsp;Synthetic images are easy to collect and annotate, yet the semantic segmentation model naively trained on synthetic data may not generalize well to real images. In this work, we introduce a conditional GAN model to close the gap between the representations of synthetic images to those of real images, thus improve the segmentation performance without laborious annotations on real image data.

![](https://github.com/heiretodemon/GAN/blob/master/Conditional%20Generative%20Adversarial%20Network%20for%20Structured%20Domain%20Adaptation/1.png)

![](https://github.com/heiretodemon/GAN/blob/master/Conditional%20Generative%20Adversarial%20Network%20for%20Structured%20Domain%20Adaptation/3.png)

## Method

###  Structure

![](https://github.com/heiretodemon/GAN/blob/master/Conditional%20Generative%20Adversarial%20Network%20for%20Structured%20Domain%20Adaptation/2.png)

- source domain -->labeled

- target domain -->unlabeled
- Goal: train a semantic segmentation model in the source domain that generalizes to the target domain

### details

- The generator is a deep residual network which takes the features with fine-grained details from the lower-level layer as input and passes them to $3X3$ convolutional filters to integrate the noise channel.
- B residual blocks are employed to learn the residual representation between the pooled features from "Conv5" for source-domain images and the target-domain representation.
- The discriminator takes the features of target domain images and the enhanced representation of source domain features as inputs and tries to distinguish them.
- Generator  $G\left(x^{s}, z ; \theta_{G}\right)=x_{\text { Convs }}^{s}+\hat{G}\left(x^{s}, z ; \theta_{G}\right)$ transforms the feature maps  $x^s$ of a synthetic image and a noise map $z$ to an adapted feature map $x^f$.
- $\hat{G}\left(x^{s}, z ; \theta_{G}\right)$ is residual representation between the Conv5 feature maps of real and synthetic image, rather than directly computing  $x^f$.
- We feed $x^f$ to a discriminator branch $D\left(x ; \theta_{D}\right)$, as well as a pixel-wise classifier branch $T\left(x ; \theta_{T}\right)$.



### Objective 

​                    $\min _{\theta_{G}, \theta_{T}} \max _{\theta_{D}} \mathcal{L}_{d}(G, D)+\alpha \mathcal{L}_{t}(G, T)$

- $\mathcal{L}_{d}$ represents the domain loss:

  ​              $\begin{aligned} \mathcal{L}_{d}(D, G)=& \mathbb{E}_{x^{t}}\left[\log D\left(x^{t} ; \theta_{D}\right)\right]+\\ & \mathbb{E}_{x^{s}, z}\left[\log \left(1-D\left(G\left(x^{s}, z ; \theta_{G}\right) ; \theta_{D}\right)\right)\right] \end{aligned}$

- we define the task loss $\mathcal{L}_{t}​$ as multinomial logistic loss: 

​                      $\begin{aligned} \mathcal{L}_{t}(G, T)=& \mathbb{E}_{x^{s}, y^{s}, z}\left[-\sum_{i=1}^{\left|I^{s}\right|} \sum_{k=1}^{K} \mathbf{1}^{y_{i}=k} \log \left(T\left(x_{i}^{s} ; \theta_{T}\right)\right)\right.\\ &-\sum_{i=1}^{\left|I^{s}\right|} \sum_{k=1}^{K} \mathbf{1}^{y_{i}=k} \log \left(T\left(G\left(x_{i}^{s}, z ; \theta_{G}\right) ; \theta_{T}\right)\right) ] \end{aligned}$

## Performance and results

### Performance 

![](https://github.com/heiretodemon/GAN/blob/master/Conditional%20Generative%20Adversarial%20Network%20for%20Structured%20Domain%20Adaptation/4.png)

### Results

![](https://github.com/heiretodemon/GAN/blob/master/Conditional%20Generative%20Adversarial%20Network%20for%20Structured%20Domain%20Adaptation/5.png)
