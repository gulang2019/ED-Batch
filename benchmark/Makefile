ED_BATCH_BASE=${PWD}/..
MODELS := $(shell find models -name '*.cc')

DYNET_LIB=-ldynet
OoC_LIB=-lOoC

# *** Mac or linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXX_FLAGS=-std=c++17 -I${ED_BATCH_BASE}/include -L${ED_BATCH_BASE}/lib -lc++ -Ofast -O2
else
  CC=g++
  CXX_FLAGS=-std=c++17 -I${ED_BATCH_BASE}/include -L${ED_BATCH_BASE}/lib -DBOOST_REGEX -lboost_regex -O2 -lpthread 
endif

# The -seq version is commented out for compatibility with master. If using the sequence-ops
# branch you can compile this program as well

all: test_block test_graph

clean:
	rm -f test_block test_graph 

test_block: ${MODELS} test_block.cc
	${CC} -o $@ $^ ${CXX_FLAGS} ${DYNET_LIB} ${OoC_LIB}

test_graph: ${MODELS} test_graph.cc
	${CC} -o $@ $^ ${CXX_FLAGS} ${DYNET_LIB} ${OoC_LIB}
