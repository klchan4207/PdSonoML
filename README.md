# Prediction of reaction rate constant on a Pd-catalyzed Sonogashira Reactions dataset

This is the code used to generate the results published in [Structure Based Descriptors and Incorporation of Domain Knowledge for Machine Learning: A case study of Pd-catalyzed Sonogashira Reactions]


# Description :

A description of the different folders is given below :
- **DATA** : this folder contains input data for descriptors and data from palladium-catalyzed Sonogashira cross-coupling reaction using trialkyl phosphine ligands reported by [Plenio *et al.*] (https://doi.org/10.1002/chem.200701418)
- **Model** : this folder contains models (DNN,GNN with MPNN layer) used in this paper
- **Train_test** : this folder contains model results  used in this paper (GNN results, DNN and Restricted Linear Regression results for Buried Volume (BV), Cone Angle (CA), Buried Volume with Cone Angle (BVCA), One-hot Encoding (ONEHOT), Multiple Fingerprint Features (MFF) ). (__evalperform contains only the template file used by above mentioned folder to generate results)


# Install requirements

Please run the following codes first to install all requirements :
```
conda create --name <env> --file requirements_conda.txt
pip install -r requirements_pip.txt
```

Please run the following codes if there is error regarding dgl configuration on tensorflow (you may change it manually in config.json under .dgl/ directory):
```
python __change_dglconfig.py
```
