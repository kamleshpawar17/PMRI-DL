# PMRI-DL

## This is code repository for the paper

Pawar K, Egan GF, Chen Z, Domain Knowledge Augmentation of Parallel MR Image Reconstruction using Deep Learning, Computerized Medical Imaging andGraphics(2021), doi:https://doi.org/10.1016/j.compmedimag.2021.101968

## Installation 
```
git clone https://github.com/kamleshpawar17/PMRI-DL.git
cd PMRI-DL
sh setup.sh
```
The ```setup.sh``` will download the data and model weights

Note that it will only download the data for Figure.9 as the data for other figures data must be donwloaded from [fastMRI](https://fastmri.med.nyu.edu/) 

## To run prediction on experimental data (Figure.9 in the paper)
```
python3 predictExp.py
```

## To run prediction for figures 3-8 use 
```
python3 predictSim.py brain filename.h5
```
first argument can be ```brain ```  or ```knee```, second argument must be path to *.h5 kspace data file dowloaded from from [fastMRI](https://fastmri.med.nyu.edu/)

## Dependencies
```
tensorflow==1.14.0
keras==2.2.4
pygrappa
```
