# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -std=c++11 -Wall
# HiGHS library path
HIGHS_PATH_LIB = /usr/local/lib
HIGHS_PATH_INCLUDE = /usr/local/include/highs

# Source files
SRCS = main.cpp
# Object files
OBJS = $(SRCS:.cpp=.o)
# Executable name
EXEC = my_program

# Rule to build executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -L$(HIGHS_PATH_LIB) -o $@ $^ -lhighs

# Rule to compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(HIGHS_PATH_INCLUDE) -c $^

# Rule to clean generated files
clean:
	rm -f $(OBJS) $(EXEC)