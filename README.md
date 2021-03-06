# SymNet
This is the matlab implementation of the proposed lightweight cascaded SPD manifold deep lerning network for SPD matrix nonlinear learning.
If you find this work is useful for your research, please kindly cite the following:
    
    R. Wang, X.-J. Wu, and J. Kittler, "SymNet: A Simple Symmetric Positive Definite Manifold Deep Learning Method for Image Set 
    Classification," in IEEE Transactions on Neural Networks and Learning Systems, 2020.  

# The folder SymNet-v1 contains three .m files:

    (1) deepmain.m is the main file, which implements the structure of SymNet-v1;
    (2) computeCov.m is constructed to compute the SPD matrices for the training and test image sets (video clips); 
    (3) fun_SymNet_Train.m is applied to implement the KDA algorithm.

# If you want to run SymNet-v1, you should:

    (1) place the four .mat files in the folder of SymNet to the folder of SymNet-v1;
    (2) run deepmain.m.
 
After a few seconds, the classification score will be output.

# Requirements
    Matlab R2019a software

# Dataset

    (1) FPHA_train_seq.mat and FPHA_train_label.mat are the training samples and the corresponding label information, respectively;
    (2) FPHA_val_seq.mat and FPHA_val_label.mat are the test samples and the corresponding label information, respectively;
    (3) This dataset is provided by \cite{FPHA}. Please kindly refer to it.

    @inproceedings{FPHA,
     title={First-person hand action benchmark with rgb-d videos and 3d hand pose annotations},
     author={Garcia-Hernando, Guillermo and Yuan, Shanxin and Baek, Seungryul and Kim, Tae-Kyun},
     booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
     pages={409--419},
     year={2018}
     }

# The code of its deep version, i.e., SymNet-v2, is in the folder of SymNet-v2.
    if you want to run SymNet-v2, you should place the four .mat files to SymNet-v2 folder,firstly. Then, run deepmain_v2.m.
    After a few seconds, the classification accuracy will be output.
    
# The Classification results of SymNet-v1 and SymNet-v2 on the FPHA dataset are listed as below:
    (1) SymNet-v1: 81.04%
    (2) SymNet-v2: 82.96%
