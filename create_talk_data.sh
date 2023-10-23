python linear_sgd.py -t 10 -s constant      -z 1  -Z uniform -i 10000 -p quasi -S inf -I random
python linear_sgd.py -t 15 -s deterministic -z 1  -Z uniform -i 10000 -p quasi -S inf -I random
python linear_sgd.py -t 15 -s adaptive      -z 10 -Z uniform -i 10000 -p quasi -S inf -I random

python linear_sgd.py -t 10 -s constant      -z 1  -Z optimal -i 10000 -p quasi -S inf -I random
python linear_sgd.py -t 15 -s deterministic -z 1  -Z optimal -i 10000 -p quasi -S inf -I random
python linear_sgd.py -t 15 -s adaptive      -z 10 -Z optimal -i 10000 -p quasi -S inf -I random
