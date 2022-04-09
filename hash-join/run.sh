#!/bin/bash
# mem = 11047
blocks=200000000

# 90% 0
# 200% 5524
# 300% 3682
# 400% 2762

# 11806
# 0 6282 8124 9044
for alloc in 0 6282 8124 9044
do
	echo occupying memory
	../src-micro/allocate.x $alloc &
	sleep 3

	for run in 1 2 3
	do
		echo prefetch
		./bin/release/bench -b 7 -R $blocks -S $blocks -x 1 -y 1 -a HJC -D 0 -L 0 -A $alloc \
		> prefetch-$run-$alloc.txt
		dmesg | tail -n 10 >> prefetch-$run-$alloc.txt
		sleep 1

		echo discard
		./bin/release/bench -b 7 -R $blocks -S $blocks -x 1 -y 1 -a HJC -D 1 -L 0 -A $alloc \
		> discard-$run-$alloc.txt
		dmesg | tail -n 10 >> discard-$run-$alloc.txt
		sleep 1

		echo discardlazy
		./bin/release/bench -b 7 -R $blocks -S $blocks -x 1 -y 1 -a HJC -D 1 -L 1 -A $alloc \
		> discardlazy-$run-$alloc.txt
		dmesg | tail -n 10 >> discardlazy-$run-$alloc.txt
		sleep 1
	done

	echo killing allocator
	kill $(pgrep allocate)
	sleep 3
done