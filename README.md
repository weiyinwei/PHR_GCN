# Personalized Hashtag Recommendation for Micro-videos, ACM MM2019
This is our Pytorch implementation for the paper:  
> Yinwei Wei, Zhiyong Cheng, Xuzheng Yu, Zhou Zhao, Lei Zhu, and Liqiang Nie(2019). Personalized Hashtag Recommendation for Micro-videos. In ACM MM`19, NICE, France,Oct. 21-25, 2019  

## Introduction
GCN_PHR is a novel graph convolutional networks (GCN) based hashtag recommendation method, which attempts to design a personalized hashtag recommendation method formicro-videos. 

## Environment Requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
- Pytorch == 1.1.0
- torch-cluster == 1.4.2
- torch-geometric == 1.2.1
- torch-scatter == 1.2.0
- torch-sparse == 0.4.0
- numpy == 1.16.0

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes.
- YFCC100M dataset  
`python train.py --l_r=0.001 --weight_decay=0.1 --batch_size=1024 --dim_latent=64 --num_workers=30 --aggr_mode='mean' --scoring_mode='cat'`
- Instagram dataset  
`python train.py --l_r=0.001 --weight_decay=0.1 --batch_size=1024 --dim_latent=64 --num_workers=30 --aggr_mode='mean' --scoring_mode='cat'` 

Some important arguments:  
- aggr_mode  
  It specifics the type of aggregation layer. Here we provide three options:  
  1. `mean` (by default) implements the mean aggregation in aggregation layer. Usage `--aggr_mode 'mean'`
  2. `max` implements the max aggregation in aggregation layer. Usage `--aggr_mode 'max'`
  3. `add` implements the sum aggregation in aggregation layer. Usage `--aggr_mode 'add'`
- `scoring_mode`:  
  It indicates the implementation of user-specific micro-video/hashtag representation. Here we provide two options:
  1. `cat`(by default) concatenates the representation of user and micro-video/hashtag, and then feed into a MLP to calculate the representations. Usage `--scoring_mode 'cat'`
  2. `fully_con` calculates the user-specific micro-video/hashtag representations with a fully connected layer. Usage `--concat 'fully_con'`

Baselines:  

  - `UTM` proposed in [User conditional hashtag prediction for images](http://www.thespermwhale.com/jaseweston/papers/imagetags.pdf), SIG KDD2015. 
  - `ConTagNet` proposed in [ConTagNet: Exploiting usercontext for image tag recommendation
](https://www.researchgate.net/publication/308855153_ConTagNet_Exploiting_User_Context_for_Image_Tag_Recommendation), ACM MM2016.[[CODE](https://github.com/vyzuer/contagnet)] 
  - `CSMN` proposed in [Attend to You: Personalized Image Captioning with Context Sequence Memory Networks](http://zpascal.net/cvpr2017/Park_Attend_to_You_CVPR_2017_paper.pdf), CVPR2017.[[CODE](https://github.com/cesc-park/attend2u)]
  - `USHM` proposed in [Separating self-expression and visual content in hashtag supervision](https://arxiv.org/abs/1711.09825), CVPR2018. 
## Dataset
We provide two processed datasets: YFCC100M and Instagram.  
- You can find the full version of dataset via [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) and raw data [Instagram]().
- We select some users and micro-videos and extract the visual, acoustic, and textual features of micro-video.

||#Micro-video|#User|#Hashtag|Visual|Acoustic|Textual|
|:-|:-|:-|:-|:-|:-|:-|
|YFCC100M|134,992|8,126|23,054|2,048|128|100|
|Instagram|48,888|2,303|12,194|2,048|128|100|

-`train.npy`  
   Train file. Each line is a user with her/his hashtag towards the micro-video: (userID, Hashtag ID and micro-video ID)  
-`val.npy`  
   Validation file. Each line is a user with her/his 1,000 negative hashtags and several positive hashtags for a micro-video: (userID, Neg_Hashtag ID, Pos_Hashtag ID, and micro-video ID)  
-`test.npy`  
   Test file. Each line is a user with her/his 1,000 negative hashtags and several positive hashtags for a micro-video: (userID, Neg_Hashtag ID, Pos_Hashtag ID, and micro-video ID) 
