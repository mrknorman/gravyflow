#OBJS specifies which files to compile as part of the project
CONFIG_IO  = ./src/config_io_test.c
TEXT_IO   = ./src/text_io_test.c

#CC specifies which compiler we're using
CC = gcc

#COMPILER_FLAGS specifies the additional compilation options we're using

INCLUDE         = -I./include -I./include/configs
COMPILER_FLAGS = -march=native -Ofast -fopenmp -std=gnu11

DEBUG_FLAG   = -g
WARNING_FLAG = -Wall -Wextra -Wconversion -pedantic

#LINKER_FLAGS specifies the libraries we're linking against
LINKER_FLAGS = -lcrypt -lpthread -ldl  -lutil -lrt -lm

#OBJ_NAME specifies the name of our exectuable
CONFIG_IO_OUT = ./bin/config_io_test
TEXT_IO_OUT   = ./bin/text_io_test

#This is the target that compiles our executable
all : $(CONFIG_IO)
	$(CC) $(CONFIG_IO) $(INCLUDE) $(COMPILER_FLAGS) $(WARNING_FLAG) $(LINKER_FLAGS) -o $(CONFIG_IO_OUT) 2> ./warnings/config.warn
	$(CC) $(TEXT_IO)   $(INCLUDE) $(COMPILER_FLAGS) $(WARNING_FLAG) $(LINKER_FLAGS) -o $(TEXT_IO_OUT)   2> ./warnings/text.warn
	./bin/config_io_test
	./bin/text_io_test

test : $(CONFIG_IO)
	$(CC) $(CONFIG_IO) $(INCLUDE) $(COMPILER_FLAGS) $(WARNING_FLAG) $(LINKER_FLAGS) -o $(CONFIG_IO_OUT) 2> ./warnings/config.warn
	$(CC) $(TEXT_IO)   $(INCLUDE) $(COMPILER_FLAGS) $(WARNING_FLAG) $(LINKER_FLAGS) -o $(TEXT_IO_OUT)   2> ./warnings/text.warn
	./bin/config_io_test
	./bin/text_io_test

debug : $(CONFIG_IO)
	$(CC) $(CONFIG_IO) $(INCLUDE) $(COMPILER_FLAGS) $(WARNING_FLAG) $(LINKER_FLAGS) $(DEBUG_FLAG) -o $(CONFIG_IO_OUT) 2> ./warnings/config.warn    
	$(CC) $(TEXT_IO)   $(INCLUDE) $(COMPILER_FLAGS) $(WARNING_FLAG) $(LINKER_FLAGS) $(DEBUG_FLAG) -o $(TEXT_IO_OUT)   2> ./warnings/text.warn
