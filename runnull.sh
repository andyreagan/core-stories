# for I in 40;
# do
#     python hierarchical-clusting-004.py $I true
#     python PCA-SVD-006.py $I true
#     python SOM-002.py $I true
# done

for I in {1..10};
do
    python PCA-SVD-006.py 40 true "-$I"
    python hierarchical-clusting-004.py 40 true "-$I"
    python SOM-002.py 40 true "-$I"
done
