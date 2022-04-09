#!/bin/bash
# MEM = 10741 = 10240 + x
blocks=640

# 90%  0
# 200% 5371
# 300% 3580
# 400% 2685

# 11806
# 0 6435 8226 9121
for alloc in 0
do
	echo occupying memory
	../src-micro/allocate.x $alloc &
	sleep 3

	for run in 1 2 3
	do
		echo sort prefetch
		./sort.x -s $blocks --passes 1 --uvm --style 1 > prefetch-$run-$alloc.txt
		dmesg | tail -n 10 >> prefetch-$run-$alloc.txt
		sleep 1

		echo sort discard
		./sort.x -s $blocks --passes 1 --uvm --discard --style 1 > discard-$run-$alloc.txt
		dmesg | tail -n 10 >> discard-$run-$alloc.txt
		sleep 1

		echo sort discard lazy
		./sort.x -s $blocks --passes 1 --uvm --discard --lazy --style 1 > discardlazy-$run-$alloc.txt
		dmesg | tail -n 10 >> discardlazy-$run-$alloc.txt
		sleep 1
	done

	echo killing allocator
	kill $(pgrep allocate)
	sleep 3
done

for alloc in 6435 8226 9121
do
	echo occupying memory
	../src-micro/allocate.x $alloc &
	sleep 3

	for run in 1 2 3
	do
		echo sort prefetch
		./sort.x -s $blocks --passes 1 --uvm  --style 0 > prefetch-$run-$alloc.txt
		dmesg | tail -n 10 >> prefetch-$run-$alloc.txt
		sleep 1

		echo sort discard
		./sort.x -s $blocks --passes 1 --uvm --discard  --style 0 > discard-$run-$alloc.txt
		dmesg | tail -n 10 >> discard-$run-$alloc.txt
		sleep 1

		echo sort discard lazy
		./sort.x -s $blocks --passes 1 --uvm --discard --lazy  --style 0 > discardlazy-$run-$alloc.txt
		dmesg | tail -n 10 >> discardlazy-$run-$alloc.txt
		sleep 1
	done

	echo killing allocator
	kill $(pgrep allocate)
	sleep 3
done