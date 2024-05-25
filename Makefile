all:
	nvcc -o libdsalgo.so src/main.cu -Llib -ldsalgo -I./src -std=c++11

clean:
	rm -f libdsalgo.so
