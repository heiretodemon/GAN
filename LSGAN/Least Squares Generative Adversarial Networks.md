[TOC]

# Least Squares Generative Adversarial Networks

[paper link](http://openaccess.thecvf.com/content_iccv_2017/html/Mao_Least_Squares_Generative_ICCV_2017_paper.html)

## GANs' problem

```latex
May lead to the vanishing gradients problem during the learning process.
```

## LSGANs'  advantages

- Generate higher quality images than regular GANs
- Perform more stable during the learning process.

## Idea
### Quality
![](https://github.com/heiretodemon/GAN/blob/master/LSGAN/1.png)

```latex
The idea is simple yet powerful:the least squares loss function is able to move the fake samples toward the decision boundary, because the least squares loss function penalizes samples that lie in a long way on the correct side of the decision boundary. As figure shows, the least squares loss function will penalize the fake samples and pull them toward the decision boundary even though they are correctly classified.
```

### Stability of learning process

- Instability of GANs learning is partially caused by the objective function which suffers from vanishing gradients.
- LSGAN penalize samples based on their distances to the decision boundary which can relieve this problem.

## Method
&emsp; The objective functions:

​                     $\begin{aligned} \min _{D} V_{\mathrm{LSGAN}}(D) &=\frac{1}{2} \mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}\left[(D(\boldsymbol{x})-b)^{2}\right] \\ &+\frac{1}{2} \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}\left[(D(G(\boldsymbol{z}))-a)^{2}\right] \end{aligned}$

​                      $\min _{G} V_{\text { tSCAN }}(G)=\frac{1}{2} \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}\left[(D(G(\boldsymbol{z}))-c)^{2}\right]$

&emsp; where $a$ and $b$ are the labels for fake data and real data, and $c$ denotes the value that $G$ wants $D$ to believe for fake data.

### Relation to Pearson $x^2$ Divergence

&emsp; Extension of objective funcitions

​                          $\begin{aligned} \min _{D} V_{\mathrm{LSGAN}}(D) &=\frac{1}{2} \mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}\left[(D(\boldsymbol{x})-b)^{2}\right] \\ &+\frac{1}{2} \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}\left[(D(G(\boldsymbol{z}))-a)^{2}\right] \end{aligned}$

​                          $\begin{aligned} \min _{G} V_{\mathrm{LSGAN}}(G) &=\frac{1}{2} \mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}\left[(D(\boldsymbol{x})-c)^{2}\right] \\ &+\frac{1}{2} \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}\left[(D(G(\boldsymbol{z}))-c)^{2}\right] \end{aligned}$

&emsp; When $G$ is fixed, the optimal discriminator $D$ is:

​                           $D^{*}(\boldsymbol{x})=\frac{b p_{\mathrm{data}}(\boldsymbol{x})+a p_{g}(\boldsymbol{x})}{p_{\mathrm{data}}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}​$

&emsp; Then we can reformulate $V_{LSGAN}​$ ：

![](https://github.com/heiretodemon/GAN/blob/master/LSGAN/2.png)

&emsp; Setting $a=-1,b=1, c=0$ is OK, and $a=0, c=b=-1$ also show similar performance

### Model architecture

![](https://github.com/heiretodemon/GAN/blob/master/LSGAN/3.png)

Motivated by the VGG model and DCGAN.
