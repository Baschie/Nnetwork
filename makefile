objects = mnistnet.o nnetwork.o reader.o matrix.o

mnistnet : $(objects)
	cc -o mnistnet $(objects) -lm

mnistnet.o : nnetwork.h reader.h
reader.o : reader.h nnetwork.h matrix.h
nnetwork.o: nnetwork.h matrix.h
matrix.o : matrix.h

.PHONY : clean
clean :
	rm mnistnet $(objects)
