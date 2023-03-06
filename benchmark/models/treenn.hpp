#pragma once
#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <iostream>
#include <assert.h>

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include <dynet/lstm.h>
#include <dynet/param-init.h>

#include "model.hpp"

namespace OoCTest
{

  class Tree
  {
  public:
    Tree(const std::string &label, std::vector<Tree *> children = std::vector<Tree *>())
        : label(label), children(children) {}
    ~Tree()
    {
      // for (auto child : children)
      //   delete child;
    }

    static Tree *from_sexpr(const std::string &str);

    static std::vector<std::string> tokenize_sexpr(const std::string &s);

    static Tree *within_bracket(std::vector<std::string>::const_iterator &tokit);

    void nonterms(std::vector<Tree *> &ret)
    {
      if (!isleaf())
      {
        ret.push_back(this);
        for (Tree *child : children)
          child->nonterms(ret);
      }
    }

    bool isleaf() const { return children.size() == 0; }

    void make_vocab(dynet::Dict &nonterm_voc, dynet::Dict &term_voc)
    {
      (isleaf() ? term_voc : nonterm_voc).convert(label);
      for (Tree *tr : children)
        tr->make_vocab(nonterm_voc, term_voc);
    }

    std::string label;
    std::vector<Tree *> children;
    dynet::Expression expr;
  };

  class TreeBuilder
  {
  public:
    virtual void start_graph(dynet::ComputationGraph& cg) = 0;
    virtual void expr_for_tree(Tree* tree, bool decorate = false) = 0;
    virtual void reset() {};
    virtual void get_visited(std::vector<Tree*>& __visited) = 0;
  };

  class TreeLSTMBuilder: public TreeBuilder
  {
  public:
    TreeLSTMBuilder(dynet::ParameterCollection &model, dynet::Dict &word_vocab, unsigned wdim, unsigned hdim, bool double_typed = false);

    void start_graph(dynet::ComputationGraph &c) override
    {
      cg = &c;
      if (!dynet::blocked){
        cg_WS.resize(WS.size());
        for (size_t i = 0; i < WS.size(); ++i)
          cg_WS[i] = dynet::parameter(*cg, WS[i]);
      }
    }

    inline void expr_for_tree(Tree* tree, bool decorate) override{
      expr_for_tree_impl(tree, decorate);
    }

    std::pair<dynet::Expression, dynet::Expression> expr_for_tree_impl(Tree* tree, bool decorate = false);

    void define_bb();

    void reset() override {visited.clear();}

    void get_visited(std::vector<Tree*>& __visited) override{
      for (auto& kv: visited) __visited.push_back(kv.first);
      return; 
    }

    dynet::ParameterCollection &model;
    dynet::Dict &word_vocab;
    unsigned wdim, hdim;
    std::vector<dynet::Parameter> WS;
    dynet::LookupParameter E;

    dynet::ComputationGraph *cg;
    std::vector<dynet::Expression> cg_WS;
    std::unordered_map<Tree*, std::pair<dynet::Expression, dynet::Expression> > visited;
    OoC::Block input_block, internal_block, internal_block1;
    bool double_typed;
  };
  
  class TreeGRUBuilder: public TreeBuilder
  {
  public:
    TreeGRUBuilder(
      dynet::ParameterCollection &model, 
      dynet::Dict &word_vocab, 
      unsigned wdim, 
      unsigned hdim);

    void start_graph(dynet::ComputationGraph &c) override
    {
      cg = &c;
      if (!dynet::blocked){
        cg_WS.resize(WS.size());
        for (size_t i = 0; i < WS.size(); ++i)
          cg_WS[i] = dynet::parameter(*cg, WS[i]);
      }
    }

    inline void expr_for_tree(Tree* tree, bool decorate) override {
      expr_for_tree_impl(tree, decorate); 
    }

    dynet::Expression expr_for_tree_impl(Tree* tree, bool decorate = false);

    void define_bb();

    void reset() override {visited.clear();}

    void get_visited(std::vector<Tree*>& __visited) override{
      for (auto& kv: visited) __visited.push_back(kv.first);
      return; 
    }

    dynet::ParameterCollection &model;
    dynet::Dict &word_vocab;
    unsigned wdim, hdim;
    std::vector<dynet::Parameter> WS;
    dynet::LookupParameter E;

    dynet::ComputationGraph *cg;
    std::vector<dynet::Expression> cg_WS;
    std::unordered_map<Tree*, dynet::Expression > visited;
    OoC::Block input_block, internal_block;
  };

  class Treenn : public Model
  {
  public:
    enum type_t {
      NORMAL, // the tree lstm
      DOUBLETYPE,
      PERFECT,
      GRID, 
      GRU
    };
    Treenn(dynet::ParameterCollection& model, int wembed_size, int hidden_size, type_t type = NORMAL, int depth_min = 5, int depth_max = 8);
    dynet::Expression build_graph(dynet::ComputationGraph &cg, int batch_size) override;
    void reset() override {data_idx = 0;}
    ~Treenn();

  private:
    TreeBuilder *builder = nullptr;
    dynet::ParameterCollection& model;
    dynet::Parameter W_param;
    std::vector<Tree *> train;
    int data_idx = 0;
    dynet::Dict nonterm_voc, term_voc;
  };

  class MVRNN: public Model {
  public:
    MVRNN(dynet::ParameterCollection& model, unsigned hidden_size);
    dynet::Expression build_graph(dynet::ComputationGraph &cg, int batch_size) override;
    void reset() override {data_idx = 0;}

  private:
    std::pair<dynet::Expression, dynet::Expression> 
      expr_for_tree(Tree &tree, bool decorate = false);

    dynet::ParameterCollection& model;
    dynet::Parameter WO;
    std::vector<dynet::Parameter> WS;
    std::vector<dynet::Expression> ws;
    std::vector<dynet::LookupParameter> LWS;
    std::vector<Tree *> train;
    int data_idx = 0;
    dynet::Dict nonterm_voc, term_voc;
    OoC::Block block;
    dynet::ComputationGraph* _cg;
  };
} 