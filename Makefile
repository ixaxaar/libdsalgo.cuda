CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -I./src -Xcompiler -fPIC -std=c++11

SRCDIR = src
LIBDIR = lib
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cu=$(SRCDIR)/%.o)
LIBRARY = $(LIBDIR)/libdsalgo.so

all: clean $(LIBRARY)

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

$(LIBRARY): $(OBJECTS)
	$(NVCC) -shared -o $@ $^

clean:
	rm -f $(SRCDIR)/*.o $(LIBRARY)

.PHONY: all clean
