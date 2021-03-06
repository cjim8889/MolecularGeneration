# MolecularGeneration

## About this Repo

This is the main code base for the Research Project: __Discrete Time E(n) Equivariant Normalising Flows for Molecular Generation__ at Imperial. This project aims to improve on the results of E(n) Equivariant Normalising Flows from Satorras et al. 2021.  

This code base is breaking and constantly changing, excersise caution when using it!  

Currently, two types of flow are implemented, namely GraphNVP-MoFlow, and [ArgmaxFlow](https://arxiv.org/abs/2102.05379).

This repo is heavily inspired by the original ArgmaxFlow and SurVAE papers by Hoogebooml et al. and Nielsen et al. 

The use of WanDB is assumed and can not be disabled at the moment. 


## Experiments

To run the Argmax Flow Edge Generation Experiment, Use the following command:
```
python train.py --type argmaxadj --batch_size 256 --epochs 200
```


To run the Argmax Flow Edge Generation Experiment V2 with hydrogen atoms removed and triangular tensor, Use the following command:
```
python train.py --type argmaxadjv2 --batch_size 128 --epochs 200 --block_length 6
```

