# comp9417Project-KNN
This repository is created for comp9417 project 2.2 ---- KNN and managed by Shengnian Yang &amp; Onur Alp Bicer

Final.py is the source code for this project. It includes methods for doing knn calculations and dataset unpacking.
To use the program, use loocv method with the following parameters

Dataset is 'Custom', 'Ionosphere', 'Automobile' # If custom dataset is not present in the current directory when Custom option is used, it will be created with following settings
Type is 'classification' or 'regression'
k is k for nearest neighbors
Distance is 'Euclidean' or 'Manhattan'
Weighted is True or False
N is number of samples in the custom dataset (only active when custom option is used AND dataset doesn't exist in current directory)

To see the Bayes error, uncomment line 312
or paste the following line into main
print(bayeserror())
