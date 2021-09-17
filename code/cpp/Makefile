# Set the compiler
CXX=clang++
CXXFLAGS=-std=c++11 -I./libs/eigen/
WARNINGS=-Wall

main: NN.o

NN.o:./NN.cpp ./NN.hpp
	$(CXX) $(CXXFLAGS) $(WARNINGS) -c ./NN.cpp

clean:
	rm main *.o
