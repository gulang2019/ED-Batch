#pragma once
#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <string>

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include <dynet/param-init.h>
#include <dynet/ooc-computation_graph.h>

#include "model.hpp"
#include "utils.h"

namespace OoCTest{
    class NdLSTMBuilder {
    public: 
        NdLSTMBuilder(dynet::ParameterCollection & model, unsigned wdim, unsigned hidden_size, unsigned vocab_size, int ndims);
        
        std::pair<dynet::Expression, dynet::Expression> build(const std::vector<int>& instance);

        dynet::Expression build_graph(const std::vector<int> &dims);

        void start_graph(dynet::ComputationGraph& cg);
    private:
        OoC::Block lstm_cell;
        int vocab_size, ndim;
        dynet::LookupParameter word_embed;
        dynet::LookupParameter H0, C0;
        std::vector<dynet::Parameter> WS;
        std::vector<dynet::Expression> ws;
        std::vector<int> dims;
        dynet::Parameter WH, BH, WC, BC;
        dynet::Expression wh, bh, wc, bc;
        dynet::ComputationGraph * cg = nullptr;
        OoC::TupleDict<std::pair<dynet::Expression, dynet::Expression> > visited;
    };

    class NdLSTM: public Model {
    public: 
        NdLSTM(
            dynet::ParameterCollection& model, 
            unsigned wembed_size, 
            unsigned hidden_size, 
            unsigned vocab_size, 
            std::initializer_list<std::pair<int, int>> dims);
        void reset() override {data_idx = 0;}
        dynet::Expression build_graph(dynet::ComputationGraph & cg, int batch_size) override;
    private: 
        NdLSTMBuilder * builder;
        std::vector<std::vector<int> > data;
        dynet::ParameterCollection& model;
        int data_idx = 0;
    };
} // namespace OoCTest 