# Use nvcc for CUDA files
NVCC = nvcc
FLAGS = -DDEBUG

# Final executable
TARGET = nbody

# Source files
SRCS = nbody.cu compute.cu kernels.cu
OBJS = $(SRCS:.cu=.o)

# Build executable
$(TARGET): $(OBJS)
	$(NVCC) $(FLAGS) $(OBJS) -o $(TARGET)

# Compile .cu → .o
%.o: %.cu
	$(NVCC) $(FLAGS) -c $< -o $@

# Clean
clean:
	rm -f *.o $(TARGET)

