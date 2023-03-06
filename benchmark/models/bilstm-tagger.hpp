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
    class BilstmTagger: public Model {
    public:
        BilstmTagger(dynet::ParameterCollection& model, unsigned layers, unsigned cembed_dim, unsigned wembed_dim, unsigned hidden_dim, unsigned mlp_dim, bool withchar = false);
        dynet::Expression build_graph(dynet::ComputationGraph& cg, 
        int batch_size) override;
        void reset() override {data_idx = 0;}
    
    private:
        dynet::ParameterCollection& model;
        int data_idx = 0;
        int layers;
        std::vector<std::pair<std::vector<std::string>, std::vector<std::string> > > train;
        dynet::Dict wv, cv, tv;
        std::unordered_map<std::string,int> wc;
        dynet::LookupParameter word_lookup, char_lookup;
        std::vector<dynet::Parameter> h0_c_p, h0_w_p;
        dynet::Parameter p_t1, pH, pO;
        dynet::VanillaLSTMBuilder fwdRNN, bwdRNN, cFwdRNN, cBwdRNN;
        std::vector<dynet::Expression> h0_c, h0_w;
        dynet::Expression H, O;
        OoC::Block calc_loss_block;
        bool withchar;
        
        dynet::Expression word_rep(dynet::ComputationGraph & cg, const std::string & w);
        std::pair<std::vector<dynet::Expression>, std::vector<dynet::Expression> >  build_tagging_graph(dynet::ComputationGraph & cg, const std::vector<std::string> & words);
        void sent_loss(dynet::ComputationGraph & cg, std::vector<std::string> & words, std::vector<std::string> & tags, std::vector<dynet::Expression> & losses);
        void new_graph(dynet::ComputationGraph & cg);
    };


} // namespace OoCTest