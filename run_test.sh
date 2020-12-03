
for i in {16..30}
do
    echo $i
    # python src/main.py Matlab$i
    python src/evaluate.py Matlab$i > stat/result$i.txt
done