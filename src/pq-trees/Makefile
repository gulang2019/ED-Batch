CXX ?= g++
CXXFLAGS = -std=c++14 -Wall -g

all: pqtest fuzztest

pqtest: pqnode.cc pqtest.cc pqtree.cc ext-pqtree.cc ext-pqtree.h
	$(CXX) $(CXXFLAGS) $^ -o $@ 

fuzztest: pqnode.cc fuzztest.cc pqtree.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ 

pressure_test: pqnode.cc pqtree.cc ext-pqtree.cc pressure_test.cc 
	$(CXX) $(CXXFLAGS) $^ -o $@ 