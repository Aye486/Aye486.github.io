---
title: 激活函数(Sigmoid/ReLU/LeakyReLU/PReLU/ELU)
date: 2024-10-29 09:34:00 +0800
categories: [笔记]
tags: [激活函数]
pin: false
author: Aye486

toc: true
comments: true
typora-root-url: ../../Aye486.github.io
math: false
mermaid: true



---

# 激活函数(Sigmoid/ReLU/LeakyReLU/PReLU/ELU)



> [Intro to Optimization in Deep Learning: Vanishing Gradients and Choosing the Right Activation Function](https://link.zhihu.com/?target=https%3A//blog.paperspace.com/vanishing-gradients-activation-function/), 

```python3
文章大纲
1. Sigmoid 和梯度消失(Vanishing Gradients)
    1.1 梯度消失是如何发生的？
    1.2 饱和神经元(Saturated Neurons)
2. ReLU 和神经元“死亡”(dying ReLU problem)
    2.1 ReLU可以解决梯度消失问题
    2.2 单侧饱和
    2.3 神经元“死亡”(dying ReLU problem)
    2.4 梯度更新方向的锯齿路径
3. LeakyReLU和PReLU
    3.1 LeakyReLU可以解决神经元”死亡“问题
4. 集大成者ELU(Exponential Linear Unit)
5. 如何选择合适的激活函数？
```

深度学习算法之前的机器学习算法，并不需要对训练数据作[概率统计](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=概率统计&zhida_source=entity)上的假设；但为了让深度学习算法有更好的性能，需要满足的关键要素之一，就是：网络的输入数据服从特定的分布：

1. 数据分布应该是[零均值化](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=零均值化&zhida_source=entity)的，即：通过该分布计算得到的均值约等于0。非零均值化的分布可能导致梯度消失和训练抖动。
2. 更进一步，数据分布最好是正态分布。非正态分布可能导致算法[过拟合](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=过拟合&zhida_source=entity)。
3. 另外，训练过程中，面对不同的数据batch时，[神经网络](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=神经网络&zhida_source=entity)每一层的各自的输入数据分布应自始至终保持一致，未能保持一致的现象叫做Internal Covaraite Shift，会影响训练过程。

本文将覆盖问题1和问题2，并分析如何采用合适的激活函数解决问题；最后提出一些普适性的选择激活函数的建议。至于问题3，则更多的与Batch Normalization相关。

## 1. Sigmoid 和梯度消失(Vanishing Gradients)

### 1.1 梯度消失是如何发生的？

梯度消失是一个老生长谈的话题了，我们先通过一个最简单的网络回顾下梯度消失是如何发生的，该网络由4个神经元 线性组成，神经元的激活函数都为Sigmoid。

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-1d48853107f5e848214e9078bbf8441d_b.jpg)

Sigmoid的函数图像和Sigmoid的[梯度函数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=梯度函数&zhida_source=entity)图像分别如下，从图像可以看出，函数两个边缘的梯度约为0，梯度的取值范围为(0,0.25)。：

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-ad24d463184228f488c10e421ba8fb5a_b.jpg)

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-e7227cc78bfff6e2a975f926e6be0f6a_b.jpg)

当我们求激活函数输出相对于权重参数w的偏导时，Sigmoid函数的梯度是[表达式](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=表达式&zhida_source=entity)中的一个[乘法因子](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=乘法因子&zhida_source=entity)。

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-1d5ac0a5bf35411f3e4e16c60cb3a0cb_b.jpg)

*回到上文拥有4个神经元的网络上来，运用[链式求导法则](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=链式求导法则&zhida_source=entity)，得到Loss函数相对于a神经元的输出值的[偏导](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=2&q=偏导&zhida_source=entity)表达式如下：*

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-927a51cf401146e8ba0481d8ed45a3c7_b.jpg)

*因为每个神经元(a/b/c/d)都是[复合函数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=复合函数&zhida_source=entity)，所以上面式子中的每一项都可以更进一步展开，以d对c的导数举例，展开如下，可以看到式子的中间项是[Sigmoid函数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=2&q=Sigmoid函数&zhida_source=entity)的梯度。那么拥有4个神经元的网络的[Loss函数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=2&q=Loss函数&zhida_source=entity)相对于第一层神经元a的偏导表达式中就包含4个Sigmoid梯度的乘积。而实际的[神经网络层数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=神经网络层数&zhida_source=entity)少则数十多则数百，这么多范围在(0,0.25)的数的乘积，将会是一个非常小的数字。而[梯度下降法](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=梯度下降法&zhida_source=entity)更新参数完全依赖于梯度值，极小的梯度无法让参数得到有效更新，即使有微小的更新，浅层和深层网络参数的更新速率也相差巨大。该现象就称为“**梯度消失(Vanishing Gradients)**”*

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-d8eb6408a81cebd0384d47a2bb71dbf2_b.png)

### 1.2 饱和神经元(Saturated Neurons)

饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入Sigmoid的值特别大或特别小，对应的梯度约等于0，即使从上一步传导来的梯度较大，该神经元权重(w)和偏置(bias)的梯度也会趋近于0，导致参数无法得到有效更新。

## 2. ReLU 和神经元“死亡”(dying ReLU problem)

### 2.1 ReLU可以解决梯度消失问题

ReLU激活函数的提出 就是为了解决梯度消失问题，LSTMs也可用于解决梯度消失问题(但仅限于RNN模型)。ReLU的梯度只可以取两个值：0或1，当输入小于0时，梯度为0；当输入大于0时，梯度为1。好处就是：ReLU的梯度的连乘不会收敛到0 ，连乘的结果也只可以取两个值：0或1 ，如果值为1 ，梯度保持值不变进行前向传播；如果值为0 ,梯度从该位置停止前向传播。Sigmoid和ReLU函数对比如下：

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-a52952fbbe0939a4c6d000f782e54a77_b.jpg)

### 2.2 单侧饱和

Simoid函数是双侧饱和的，意思是朝着正负两个方向，函数值都会饱和；但ReLU函数是单侧饱和的，意思是只有朝着负方向，函数值才会饱和。严格意义上来说，将ReLU函数值为0的部分称作饱和是不正确的(饱和应该是取值趋近于0)，但效果和饱和是一样的。单侧饱和有什么好处？

**让我们把神经元想象为检测某种特定特征的开关**，高层神经元负责检测高级的/抽象的特征(有着更丰富的[语义信息](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=语义信息&zhida_source=entity))，例如眼睛或者轮胎；[低层神经元](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=低层神经元&zhida_source=entity)负责检测低级的/具象的特征，例如曲线或者边缘。**当开关处于开启状态，说明在输入范围内检测到了对应的特征，且正值越大代表特征越明显**。加入某个神经元负责检测边缘，则正值越大代表边缘区分越明显(sharp)。那么负值越小代表什么意思呢？直觉上来说，用负值代表检测特征的缺失是合理的，但用负值的大小代表缺失的程度就不太合理，难道缺失也有程度吗？

假设一个负责检测边缘的神经元，激活值为10相对于激活值为5来说，检测到的边缘区分地更明显；但激活值-10相对于-5来说就没有意义了，因为低于0的激活值都代表没有检测到边缘。所以用一个[常量值](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=常量值&zhida_source=entity)0来表示检测不到特征是更方便合理的，像ReLU这样单侧饱和的神经元就满足要求。

**单侧饱和还能使得神经元对于噪声干扰更具[鲁棒性](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=鲁棒性&zhida_source=entity)**。假设一个双侧都不饱和的神经元，正侧的不饱和导致神经元正值的取值各不相同，这是我们所希望的，因为正值的大小代表了检测[特征信号](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=特征信号&zhida_source=entity)的强弱。但负值的大小引入了背景噪声或者其他特征的信息，这会给后续的神经元带来无用的干扰信息；且可能导致神经元之间的相关性，相关性(重复信息)是我们所不希望的。例如检测直线的神经元和检测曲线的神经元可能有负相关性。在负值区域单侧饱和的神经元则不会有上述问题，噪声的程度大小被饱和区域都截断为0,避免了无用信息的干扰。

使用ReLU激活函数在计算上也是高效的。相对于Sigmoid函数梯度的计算，ReLU[函数梯度](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=2&q=函数梯度&zhida_source=entity)取值只有0或1。且ReLU将负值截断为0 ，为网络引入了稀疏性，进一步提升了计算高效性。

### 2.3 神经元“死亡”(dying ReLU problem)

但ReLU也有缺点，尽管稀疏性可以提升计算高效性，但同样可能阻碍训练过程。通常，激活函数的输入值有一项偏置项(bias)，假设bias变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。

### 2.4 梯度更新方向的锯齿路径

无论ReLU的输入值是什么范围，输出值总是非负的，这也是一个缺点。采用ReLU激活函数的网络第n层的激活值的表达式为：

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-73a32cfd7580a53f4f182a3653436555_b.jpg)

[损失函数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=损失函数&zhida_source=entity)相对于参数的梯度表达式为：

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-fdb8b478993344140893cd652e693149_b.png)

其中第二项[指示函数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=指示函数&zhida_source=entity)(指示函数输入大于0,输出为1;否则,输出为0)。第三项是该层的输入(即上一层ReLU的输出)，值范围是非负的。所以对于该层所有的w参数，梯度的符号都是一样的。这会带来什么问题？梯度的符号决定了参数的更新方向，因为该层所有w参数的梯度符号相同，所以在一次更新中，该层的w参数要么一起增大，要么一起减小。但理想的参数的更新情况可能是：一次更新时，该层一部分[w参数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=4&q=w参数&zhida_source=entity)增大，而另一部分w参数减小。使用ReLU无法做到。

假设训练过程中的某一时刻：一部分参数w1需要减小，以到达理想的参数范围内；而另一部分参数w2却需要增大。然而此次[迭代计算](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=迭代计算&zhida_source=entity)得到的梯度符号一致，参数更新后都将减小(满足w1不满足w2)；下次迭代时，计算得到的梯度符号仍然一致但方向相反，参数更新后都将增大(满足w2不满足w1)。这将导致参数[更新过程](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=更新过程&zhida_source=entity)中，方向的锯齿问题，以下图理解更直观：从原地到达最优点，一部分参数需要减小(w1,以y轴表示)，另一部分参数需要增大(w2,以x轴表示)，但每次更新时，w1/w2 必须同时增大或减小，造成了更新方向的锯齿问题，减缓了训练过程。

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-81eaefba57451b08d4a5f391d4f88d35_b.jpg)

## 3. LeakyReLU和PReLU

### 3.1 LeakyReLU可以解决神经元”死亡“问题

LeakyReLU的提出就是为了解决神经元”死亡“问题，LeakyReLU与ReLU很相似，仅在输入小于0的部分有差别，ReLU输入小于0的部分值都为0，而LeakyReLU输入小于0的部分，值为负，且有微小的梯度。函数图像如下图：

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-9b1fc63cf1058e5543285494fa26a4c1_b.jpg)

实际中，LeakyReLU的α取值一般为0.01。使用LeakyReLU的好处就是：在反向传播过程中，对于LeakyReLU激活函数输入小于零的部分，也可以计算得到梯度(而不是像ReLU一样值为0)，这样就避免了上述[梯度方向](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=梯度方向&zhida_source=entity)锯齿问题。

超参数α的取值也已经被很多[实验研究](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=实验研究&zhida_source=entity)过，有一种取值方法是 对α随机取值，α的分布满足均值为0,[标准差](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=标准差&zhida_source=entity)为1的正态分布，该方法叫做随机LeakyReLU(Randomized LeakyReLU)。原论文指出随机LeakyReLU相比LeakyReLU能得更好的结果，且给出了参数α的经验值1/5.5(好于0.01)。至于为什么随机LeakyReLU能取得更好的结果，解释之一就是随机LeakyReLU小于0部分的[随机梯度](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=随机梯度&zhida_source=entity)，为[优化方法](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=优化方法&zhida_source=entity)引入了随机性，这些[随机噪声](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=随机噪声&zhida_source=entity)可以帮助参数取值跳出局部最优和鞍点，这部分内容可能需要一整篇文章来阐述。正是由于α的取值至关重要，人们不满足与随机取样α，有论文将α作为了需要学习的参数，该激活函数为PReLU(Parametrized ReLU)。

### 4. 集大成者ELU(Exponential Linear Unit)

通过上述的讨论可以看出，理想的激活函数应满足两个条件：

1. 输出的分布是零均值的，可以加快训练速度。
2. 激活函数是单侧饱和的，可以更好的收敛。

LeakyReLU和PReLU满足第1个条件，不满足第2个条件；而ReLU满足第2个条件，不满足第1个条件。两个条件都满足的激活函数为ELU(Exponential Linear Unit)，函数图像如下图：

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-fa5b4490dc4a7f698543f9d37e28b6b1_b.jpg)

表达式为：

![img](/assets/blog_res/2021-03-30-hello-world%20(copy).assets/v2-9ac3683b1411272bbdad96d9aa50c39a_b.jpg)

输入大于0部分的梯度为1,输入小于0的部分无限趋近于-α，超参数取值一般为1.

## 5. 如何选择合适的激活函数？

1. 先试试ReLU的效果如何。尽管我们指出了ReLU的一些缺点，但很多人使用ReLU取得了很好的效果。根据”[奥卡姆剃刀原理](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=奥卡姆剃刀原理&zhida_source=entity)“，如无必要，勿增实体，也就是优先选择最简单的。ReLU相较于其他激活函数，有着最低的计算代价和最简单的代码实现。
2. 如果ReLU效果不太理想，下一个建议是试试LeakyReLU或ELU。经验来看：有能力生成[零均值分布](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=零均值分布&zhida_source=entity)的激活函数，相较于其他激活函数更优。需要注意的是使用ELU的[神经网络训练](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=1&q=神经网络训练&zhida_source=entity)和推理都会更慢一些，因为需要更复杂的指数运算得到函数激活值，如果计算资源不成问题，且网络并不十分巨大，可以事实ELU；否则，最好选用LeakyReLU。LReLU和ELU都增加了需要调试的[超参数](https://zhida.zhihu.com/search?content_id=126413738&content_type=Article&match_order=3&q=超参数&zhida_source=entity)。
3. 如果有很多算力或时间，可以试着对比下包括随机ReLU和PReLU在内的所有激活函数的性能。当网络表现出过拟合时，随机ReLU可能会有帮助。而对PReLU来说，因为增加了需要学习的参数，**当且仅当有很多训练数据时才可以试试PReLU的效果**
