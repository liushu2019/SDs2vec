# SDs2vec
Code for SDs2vec

The URL of the preprint will be uploaded as soon as it passes moderation checks.

The implementation codes and data will be available here.

Please feel free to contact me (shu.liu.eq@gmail.com).

# Paper:
TBA

# Special Packages:
```
pip install numpy-quaternion
```
Note: Other packages, such as Numpy and Pandas, are needed but are not listed here.

# Usage:
```
python src/main.py --input graph/star_sign_directed.edgelist --num-walks 100 --walk-length 80 --window-size 5 --dimensions 2 --until-layer 5 --workers 10 --suffix star --OPT3 --output star_hyper
```
# Data format:
```
sourceNodeID targetNodeID sign
...
```
See example: graph/star_sign_directed.edgelist.

# Output:
```
NumberOfNodes NumberOfDimension
nodeID1 dim1 dim2 ...
nodeID2 dim1 dim2 ...
...
```
See example: emb/star_sign_directed.emb.

# Citation
```
TBA
```
