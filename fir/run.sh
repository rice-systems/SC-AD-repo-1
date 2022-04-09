#!/bin/bash
# mem = 11897
blocks=1450

# 90%  0
# 200% 5949
# 300% 3966
# 400% 2974
# page table 247


for alloc in 0 5857 7840 8832
do
	echo occupying memory
	../src-micro/allocate.x $alloc &
	sleep 3

	for run in 1 2 3
	do
		echo FIR prefetch-async
		./fir.x 1048576 $blocks 65536 0 0 > prefetch-$run-$alloc.txt
		dmesg | tail -n 10 >> prefetch-$run-$alloc.txt
		sleep 1

		echo FIR discard
		./fir.x 1048576 $blocks 65536 1 0 > discard-$run-$alloc.txt
		dmesg | tail -n 10 >> discard-$run-$alloc.txt
		sleep 1

		echo FIR discard lazy
		./fir.x 1048576 $blocks 65536 1 1 > discardlazy-$run-$alloc.txt
		dmesg | tail -n 10 >> discardlazy-$run-$alloc.txt
		sleep 1
	done

	echo killing allocator
	kill $(pgrep allocate)
	sleep 3
done