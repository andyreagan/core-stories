PYENV = pyvenv/bin/python
all: experiment control
experiment:
		# for I in 80 40 20 10;
		# do
		#     $PYENV src/hierarchical_clusting/hierarchical-clusting-004.py $I false
		#     $PYENV src/PCA_SVD/PCA-SVD-006.py $I false
		#     $PYENV src/SOM/SOM-002.py $I false
		# done
		for I in 40;
		do
				$PYENV src/hierarchical_clusting/hierarchical-clusting-004.py $I false
				$PYENV src/PCA_SVD/PCA-SVD-006.py $I false
				$PYENV src/SOM/SOM-002.py $I false
		done
control:
		# for I in 40;
		# do
		#     $PYENV src/hierarchical_clusting/hierarchical-clusting-004.py $I true
		#     $PYENV src/PCA_SVD/PCA-SVD-006.py $I true
		#     $PYENV src/SOM/SOM-002.py $I true
		# done
		for I in {1..10};
		do
				$PYENV src/PCA_SVD/PCA-SVD-006.py 40 true "-$I"
				$PYENV src/hierarchical_clusting/hierarchical-clusting-004.py 40 true "-$I"
				$PYENV src/SOM/SOM-002.py 40 true "-$I"
		done
