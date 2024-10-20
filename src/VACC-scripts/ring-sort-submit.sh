cd /users/a/r/areagan/work/2014/09-books

export ITER=30000
for I in {1..100}
do
    export I
    qsub -qworkq -V ring-sort.qsub
done
