CC=gcc
CFLAGS=-O3
LDFLAGS=-lm -lpthread

SRC_DIR=src
SRC=$(wildcard $(SRC_DIR)/*.c)
INC=$(wildcard $(SRC_DIR)/*.h)
OBJ=$(patsubst %.c,%.o,$(SRC))
TARGET=simulate

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

$(OBJ): $(SRC) $(INC)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJ)

.PHONY: var
var:
	@echo SRC: $(SRC)
	@echo INC: $(INC)
	@echo OBJ: $(OBJ)
