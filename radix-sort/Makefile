NVCC = nvcc
NVCC_FLAGS = -gencode=arch=compute_86,code=sm_86 -O3 \
-I. -I../uvmdiscard \
../uvmdiscard/uvmdiscard.cpp \
-lcuda -lcudart

TARGET=sort.x
files=*.cu *.cpp
all: $(TARGET)
	
$(TARGET):
	$(NVCC) $(NVCC_FLAGS) $(files) -o $(TARGET)
	
clean:
	rm -f *.o *~ $(TARGET)
	
