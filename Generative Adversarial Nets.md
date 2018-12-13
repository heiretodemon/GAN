[TOC]
# Generative Adversarial Nets 

[论文链接](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

# Introduction

我们提出了一个通过对抗过程来估计生成模型的新框架，其中我们同时训练两个模型：一个捕获数据分布的生成模型$G$，一个分辨生成器$G$的输出和真实数据的判别模型$D$。生成器试图产生更接近真实的数据，判别器试图更完美地分辨真实数据与生成数据。两个网络在对抗中进步，在进步中对抗。这正好对应于博弈论中的$minimax$最小最大博弈。 在任意函数$G$和$D$的空间中，存在唯一的解决方案，即$G$获得训练数据分布并且$D$等于$1/2$。 在$G$和$D$由多层感知器定义的情况下，整个系统可以用反向传播进行训练。 在训练或生成样本期间不需要任何马尔可夫链或展开的近似推理网络。 

# Related work
- restricted Boltzmann machines 
- deep Boltzmann machines                          
- Markov chain Monte Carlo
- deep belief networks
- score matching
- noise-contrastive estimation
- denoising auto-encoders and contractive autoencoders
- generative stochastic network

# Theory

定义$p_g$为生成器对样本数据分布的采样，输入噪声的先验变量$p_z{(z)}$，用$Gz;\theta_q()$来表征数据空间的映射，$G$是一个表示含有参数$\theta_g$的多层感知机可微函数。定义第二个多层感知机输出一个概率的判别器$D(x;\theta_d)$。$D(x)$表示$x$来自真实数据分布而不是$p_g$。训练$D$来最大化判定是训练样本还是$G$生成样例为真的概率。同时训练$G$来最小化$log(1-D(G(Z)))$，即$D$和$G$的训练时关于值函数$V(G,D)$的minimax博弈：

![1544628164602](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544628164602.png)

训练策略：先进行K次$D$的优化，在进行1次$G$的优化。这样，只要$G$变化得足够慢，则$D$就会一直保持最优解附近。训练过程如图1所示：

![1544669812866](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544669812866.png)

## 全局最优解证明

对于任意给定$G$，考虑最优判别器$D$。

固定$G$，最优判别器$D$是

 ![1544674372911](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544674372911.png)

证明：对于任生成器$G$,判别器$D$的训练标准就是最大化目标函数$V(G,D)$

![1544674533426](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544674533426.png)

对任意的$(a,b)\in R^2$且在$a,b\in(0,1)$，函数$f=alog(y)+blog(1-y)$在$\frac a {a+b}$时值最大。

又注意到在判别器$D$的训练目标可以看作为条件概率$P(Y=y|x)$的最大似然估计，当$y=1$时，$x$来自$p_data$，$y=0$时$x$来自$p_g$，则方程$(1)$可以变成

![1544675137708](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544675137708.png)

则当且仅当$p_g=p_data$即$D^*_G(x)=\frac 1 2$时，$C(G)$的全局最小值为$-log4$。

从$C(G)=V(D^*_G,G)$中提取这个表达式，我们可以得到

![1544675922156](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544675922156.png)

其中$KL$为[Kullback–Leibler散度](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5)。我们从之前的Jensen-Shannon散度表达式中识别出模型的分布和数据产生过程：

![1544675932615](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544675932615.png)

由于两个分布之间的Jensen-Shannon散度总是非负的，并且当两个分布相等时，值为0。因此$C^*=-log4$为$C(G)$的全局极小值，并且唯一解为$p_g = p_{data}$，即生成模型能够完美的复制数据的生成过程。

## 收敛性

命题二：如果$G$和$D$有足够的性能，对于算法1中的每一步，给定$G$时，判别器都能达到它的最优解，并且通过更新$p_g$来提高这个判别准则

![1544676713378](C:\Users\Allen\AppData\Roaming\Typora\typora-user-images\1544676713378.png)则$p_g$从收敛为$p_{data}​$。

# Advantages and disadvantages
## Disadvantages
- $p_g(x)$的表征不明确
- 训练期间，$D$和$G$必须很好地同步，尤其是为了避免“Helvetica”场景即在$x$相同时$G$丢失过多$Z$值以至于模型$p_data$多样性不足，在不更新$D$时，$D$不必过度训练。
## Advantages
- 计算性。
- 可以表征非常尖锐甚至退化的分布