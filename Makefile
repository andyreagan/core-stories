all: experiment control
experiment:
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 false
	python src/PCA_SVD/PCA-SVD-006.py 40 false
	python src/SOM/SOM-002.py 40 false
control:
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-1"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-1"
	python src/SOM/SOM-002.py 40 true "-1"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-2"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-2"
	python src/SOM/SOM-002.py 40 true "-2"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-3"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-3"
	python src/SOM/SOM-002.py 40 true "-3"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-4"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-4"
	python src/SOM/SOM-002.py 40 true "-4"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-5"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-5"
	python src/SOM/SOM-002.py 40 true "-5"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-6"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-6"
	python src/SOM/SOM-002.py 40 true "-6"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-7"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-7"
	python src/SOM/SOM-002.py 40 true "-7"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-8"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-8"
	python src/SOM/SOM-002.py 40 true "-8"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-9"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-9"
	python src/SOM/SOM-002.py 40 true "-9"
	python src/PCA_SVD/PCA-SVD-006.py 40 true "-10"
	python src/hierarchical_clustering/hierarchical-clusting-004.py 40 true "-10"
	python src/SOM/SOM-002.py 40 true "-10"
