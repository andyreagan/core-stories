for VAR in {200..490}
do
  export COUNT=$((VAR*30))
  export NUMPER=30
  echo "$COUNT $NUMPER"
  sleep 0.5
  qsub -q workq -V rundistance.qsub
done