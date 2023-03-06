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

// #include <dynet/ooc-computation_graph.h>
namespace OoCTest
{
    struct Arc
    {
        int from;
        int id;
        int label;
        bool trivial = false;
    };

    struct State
    {
        dynet::Expression h, c;
        std::vector<Arc> arcs;
        int tag;
    };

    struct DAG
    {
        std::vector<State> nodes;
    };

    class LatticeBuilder {
    public:
        virtual void start_graph(dynet::ComputationGraph &cg) = 0;
        virtual void build_graph(DAG& g, std::vector<dynet::Expression> & losses) = 0;
    };

    class LatticeLSTMBuilder: public LatticeBuilder
    {
    public:
        LatticeLSTMBuilder(dynet::ParameterCollection &model, unsigned hdim);

        void start_graph(dynet::ComputationGraph &cg) override;

        void build_graph(DAG &g, std::vector<dynet::Expression> &losses) override;

    private:
        dynet::Parameter H0, C0, WO;
        dynet::LookupParameter E_c, E_w;
        std::vector<dynet::Parameter> WS;

        dynet::ComputationGraph *_cg;
        dynet::Expression h0, c0, wo;
        std::vector<std::unique_ptr<OoC::Block>> blocks;
        std::vector<dynet::Expression> cg_WS;
    };
    
    class LatticeGRUBuilder: public LatticeBuilder
    {
    public:
        LatticeGRUBuilder(dynet::ParameterCollection &model, unsigned hdim);

        void start_graph(dynet::ComputationGraph &cg) override;

        void build_graph(DAG &g, std::vector<dynet::Expression> &losses) override;

    private:
        dynet::Parameter H0, C0, WO;
        dynet::LookupParameter E_c, E_w;
        std::vector<dynet::Parameter> WS;

        dynet::ComputationGraph *_cg;
        dynet::Expression h0, c0, wo;
        std::vector<std::unique_ptr<OoC::Block>> blocks;
        std::vector<dynet::Expression> cg_WS;
    };

    class Lattice: public Model
    {
    public:
        enum type_t{
            LSTM,
            GRU
        }type;
        Lattice(dynet::ParameterCollection& model, int hidden_size, type_t type = LSTM);
        ~Lattice() { delete builder; }
        dynet::Expression build_graph(dynet::ComputationGraph &cg, int batch_size) override;
        void reset() override { data_idx = 0; }

    private:
        dynet::ParameterCollection& model;
        int hidden_size;
        int data_idx = 0;
        LatticeBuilder *builder;
        std::vector<DAG> train;
    };

    class RandomDAG: public Model{
    public:
        RandomDAG(
            dynet::ParameterCollection& model,
            unsigned hdim, 
            unsigned vocab_size, 
            int n_state = 200, 
            int n_arc_type = 3, 
            int n_state_type = 3, 
            int n_data = 100, 
            int in_degree = 4, 
            int longest_dependency = 10);
        dynet::Expression build_graph(dynet::ComputationGraph &cg, int batch_size) override;
        void reset() override { data_idx = 0; }
        ~RandomDAG(){
            for(auto block: blocks) delete block;
        }
    private:
        void build_expr(dynet::ComputationGraph& cg, DAG& dag, std::vector<dynet::Expression>& losses);
        std::vector<DAG> data;
        int data_idx = 0;
        std::vector<OoC::Block*> blocks;
        OoC::Block output_block;
        dynet::ParameterCollection& model;
        dynet::LookupParameter L;
        std::vector<dynet::Parameter> WS;
        std::vector<dynet::Parameter> BS;
        dynet::Parameter S0;
    };

} // namespace OoCTest