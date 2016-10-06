# for I in 80 40 20 10;
# do
#     python hierarchical-clusting-004.py $I false
#     python PCA-SVD-006.py $I false
#     python SOM-002.py $I false
# done
for I in 40;
do
    python hierarchical-clusting-004.py $I false
    python PCA-SVD-006.py $I false
    python SOM-002.py $I false
done
