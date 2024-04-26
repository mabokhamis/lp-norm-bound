# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -std=c++11 -Wall -Wextra
# HiGHS library path
HIGHS_PATH = /usr/local

# Source files
SRCS = main.cpp
# Object files
OBJS = $(SRCS:.cpp=.o)
# Executable name
EXEC = my_program

# Rule to build executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -L$(HIGHS_PATH)/lib -o $@ $^ -lhighs

# Rule to compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(HIGHS_PATH)/include/highs -c $^

# Rule to clean generated files
clean:
	rm -f $(OBJS) $(EXEC)
