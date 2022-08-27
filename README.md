# Code for Rényi Fair and Private Information Bottleneck (RFPIB) and Rényi Fair Information Bottleneck (RFIB)

This is the code for RFPIB and for RFIB, from the paper "Achieving Utility, Fairness and Compactness via Tunable Information Bottleneck Measures."

Experiments can be run by running main.py. By default, experiments will run using RFPIB on the CelebA dataset. 
By changing variables at the top of the main() method in main.py, the code can be modified to run experiments on RFIB or on various baselines, 
and with other datasets including EyePACS and FairFace.

CSV files with the labels and partitions used are included but the datasets need to be downloaded before the code can run.
After downloading the datasets, name the directories "eyepacs", "celeba", and "fairface" and place them in the "data" directory.

EyePACs dataset can be found at: https://www.kaggle.com/c/diabetic-retinopathy-detection/data

CelebA dataset can be found at: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

FairFace dataset can be found at: https://github.com/joojs/fairface

