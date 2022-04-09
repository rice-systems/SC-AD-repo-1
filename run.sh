#!/bin/bash

rm -rf data
mkdir data
for bench in fir radix-sort hash-join
do
	mkdir data/$bench
	cd $bench
	bash run.sh
	cd ..
	echo $bench done
	sleep 1
	mv $bench/*txt data/$bench
done

# for bench in radix-sort
# do
# 	mkdir data/$bench
# 	cd $bench
# 	bash run2.sh
# 	cd ..
# 	echo $bench done
# 	sleep 1
# 	mv $bench/*txt data/$bench
# done