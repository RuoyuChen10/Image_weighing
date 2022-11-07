# Image Weighing

This code is the original experimental setup and may be out of date, if you need a clearer implementation please contact me and I can provide a more recent version based on `Pytorch`.

**Official Implementation of our Paper**

Ruoyu Chen, Yuliang Zhao*, Shuyu Wang, Lianjiang Li, Xiaopeng Sha, Yongliang Yang, Lianqing Liu, Guanglie Zhang, and Wen Jung Li. “Online Estimating Weight of White Pekin Duck Carcass by Computer Vision”, *Poultry Science*, to apear.

## 1. Visualization

Our approach is dedicated to achieving real-time image weighing.

![](./images/duck.gif)

## 2. Environment

We consider 2 environments, the first figure shows the work of this project, and the second figure is coming soon.

<table border-left=none border-right=none><tr>
<td width=50%><img src=images/data1.png border=none></td>
<td width=50%><img src=images/data2.png border=none></td>
</tr></table>

## 3. The Dataset

Please refer to [weight.xls](weight.xls) for labeling data. The images are included in fold [duck2](./duck2).

The statistics of the dataset are as follows:

![](./images/Fig2.png)

In the dataset, we remove the feet, because there are some noise, which will reduce the precise of the weight estimation results. A experiment is shown below:

![](./images/FigA3.png)

## 4. Reproduce the results 

The main demo is `train_cov_choose_duck_no_background_or_feet.py`, you can try:

```
python train_cov_choose_duck_no_background_or_feet.py
```

to reproduce the results. The structure of the CNN is shown below:

![](./images/Fig3.png)

## 5. Ablation Study

The input is a single channel image, where we choose the R channel, as you can see:

![](./images/FigA1.png)

R channel is salient for textual information, thus we choose the R channel.
