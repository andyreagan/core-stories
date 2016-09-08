# dead simple shell script for submitting a boatload of jobs
for COUNT in {2301..3100}
do
    export COUNT
    qsub -V -qshortq run.qsub
    sleep 1.5
done
