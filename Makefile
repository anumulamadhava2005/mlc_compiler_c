# Makefile for MLC Compiler (Verbose Mode with All Phases)

CC = gcc
CFLAGS = -Wall -Wno-unused-function -Wno-format-truncation
FLEX = flex
BISON = bison

# Targets
all: mlc_compiler

mlc_compiler: parser.tab.o lex.yy.o compiler_phases.o main.o
	$(CC) $(CFLAGS) -o mlc_compiler parser.tab.o lex.yy.o compiler_phases.o main.o

parser.tab.c parser.tab.h: parser.y
	$(BISON) -d -o parser.tab.c parser.y

parser.tab.o: parser.tab.c
	$(CC) $(CFLAGS) -c parser.tab.c

lex.yy.c: lexer.l parser.tab.h
	$(FLEX) lexer.l

lex.yy.o: lex.yy.c parser.tab.h
	$(CC) $(CFLAGS) -c lex.yy.c

compiler_phases.o: compiler_phases.c compiler_phases.h ast.h
	$(CC) $(CFLAGS) -c compiler_phases.c

main.o: main.c parser.tab.h compiler_phases.h
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f lex.yy.c lex.yy.o parser.tab.c parser.tab.h parser.tab.o *.o mlc_compiler

.PHONY: all clean
