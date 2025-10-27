# Compiler and flags
CC = gcc
CFLAGS = -Wall -g -O0 -I.

# Tools
BISON = bison
FLEX  = flex

# Files
PARSER = parser.y
LEXER  = lexer.l
MAIN   = main.c       # Your main C file, can also be parser.tab.c if you put everything in parser.y
TARGET = mlc_compiler

# Default target
all: $(TARGET)

# Build the parser and lexer, then compile
$(TARGET): parser.tab.c lex.yy.c $(MAIN)
	$(CC) $(CFLAGS) -o $(TARGET) parser.tab.c lex.yy.c $(MAIN) -lfl

# Generate parser files from Bison
parser.tab.c parser.tab.h: $(PARSER)
	$(BISON) -d --report=none $(PARSER)

# Generate lexer files from Flex
lex.yy.c: $(LEXER)
	$(FLEX) -o lex.yy.c $(LEXER)

# Clean generated files
clean:
	rm -f $(TARGET) parser.tab.c parser.tab.h lex.yy.c *.o train.py
