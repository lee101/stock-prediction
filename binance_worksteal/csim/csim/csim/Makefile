CC = gcc
CFLAGS = -O3 -march=native -fopenmp -ffast-math -shared -fPIC -Wall
TARGET = libworksteal.so
SRC = worksteal_sim.c

all: $(TARGET)

$(TARGET): $(SRC) worksteal_sim.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean
