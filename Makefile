SOURCES := main.c nn.c csv.c fread_csv_line.c
CFLAGS += -O3 -march=native -DNDEBUG
#CFLAGS += -O0 -g
LDFLAGS := -lm
CC := gcc

all: $(SOURCES)
	$(CC) $^ $(CFLAGS) -o cnn $(LDFLAGS)
