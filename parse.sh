#!/bin/bash

cd data

cd fir
for alloc in 0 5857 7840 8832
do
	for proto in prefetch discard discardlazy
	do
		rm -rf $proto-$alloc.txt
		for run in 1 2 3
		do
			cat $proto-$run-$alloc.txt | grep 'Runtime' | \
			grep -Eo '[+-]?[0-9]+([.][0-9]+)?' >> $proto-$alloc.txt
		done
		echo FIR $alloc $proto mem:
		cat $proto-2-$alloc.txt | grep fetch/evict 
	done
done
cd ..

cd radix-sort
for alloc in 0 6435 8226 9121
do
	for proto in prefetch discard discardlazy
	do
		rm -rf $proto-$alloc.txt
		for run in 1 2 3
		do
			cat $proto-$run-$alloc.txt | grep 'Runtime' | \
			grep -Eo '[+-]?[0-9]+([.][0-9]+)?' >> $proto-$alloc.txt
		done
		echo Radix-sort $alloc $proto mem:
		cat $proto-2-$alloc.txt | grep fetch/evict 
		cat $proto-2-$alloc.txt | grep faults
	done
done
cd ..

cd hash-join
for alloc in 0 6282 8124 9044
do
	for proto in prefetch discard discardlazy
	do
		rm -rf $proto-$alloc.txt
		for run in 1 2 3
		do
			cat $proto-$run-$alloc.txt | grep 'ms' | \
			grep -Eo '[+-]?[0-9]+([.][0-9]+)?' >> $proto-$alloc.txt
		done
	done
done
cd ..