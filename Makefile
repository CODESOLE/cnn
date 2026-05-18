SOURCES := main.c nn.c csv.c fread_csv_line.c
CFLAGS += -O3 -march=native # -DNDEBUG
#DCFLAGS += -O0 -g
LDFLAGS := -lm
CC := gcc

all: $(SOURCES)
	$(CC) $^ $(CFLAGS) -o cnn $(LDFLAGS)

d: $(SOURCES)
	$(CC) $^ $(DCFLAGS) -o cnn $(LDFLAGS)
