#pragma once 

#include "model.hpp"

namespace OoCTest{
    class TreennInternal: public Model {
    public:
        TreennInternal(unsigned hdim, unsigned wdim);
        dynet::Expression build_graph(dynet::ComputationGraph &cg,
                                 int batch_size = 32) override;
    private:
        OoC::Block block;
        dynet::ParameterCollection model;
        dynet::LookupParameter L;
        std::vector<dynet::Parameter> WS;
    };

    class TreennLeaf: public Model {
    public:
        TreennLeaf(unsigned hdim, unsigned wdim, unsigned vocab_size);
        dynet::Expression build_graph(dynet::ComputationGraph& cg,
            int batch_size = 32) override;
    private:
        unsigned vocab_size;
        OoC::Block block;
        dynet::ParameterCollection model;
        dynet::LookupParameter E;
        std::vector<dynet::Parameter> WS;
    };

    class LSTMCell: public Model {
    public:
        LSTMCell(unsigned hdim, unsigned wdim, unsigned vocab_size);
        dynet::Expression build_graph(dynet::ComputationGraph& cg,
            int batch_size = 32) override;
    private:
        unsigned vocab_size;
        OoC::Block block;
        dynet::ParameterCollection model;
        dynet::LookupParameter word_embed;
        std::vector<dynet::Parameter> WS;
    };

    class GRUCell: public Model {
    public:
        GRUCell(unsigned hdim);
        dynet::Expression build_graph(dynet::ComputationGraph & cg, 
            int batch_size) override;
    private:
        int hdim;
        OoC::Block block;
        dynet::ParameterCollection model;
        dynet::LookupParameter E;
        std::vector<dynet::Parameter> WS;
    };

    class NChildGRU: public Model {
    public:
        NChildGRU(unsigned hdim, int n_child = 1);
        dynet::Expression build_graph(dynet::ComputationGraph & cg, 
            int batch_size) override;
    private:
        int hdim, n_child;
        OoC::Block block;
        dynet::ParameterCollection model;
        dynet::LookupParameter E;
        std::vector<dynet::Parameter> WS;
    };

    class MVCell: public Model {
    public:
        MVCell(unsigned hdim);
        dynet::Expression build_graph(dynet::ComputationGraph & cg, 
            int batch_size) override;
    private:
        int hdim;
        OoC::Block block;
        dynet::ParameterCollection model;
        std::vector<dynet::Parameter> WS;
    };
} // namespace OoCTest