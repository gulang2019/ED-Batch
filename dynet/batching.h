#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <set>
#include <algorithm>
#include <chrono>
#include <assert.h>
#include <queue>
#include "dynet/dynet.h"
using namespace std;

struct Timer {
  string type;
  Timer(const char * t = "DEFAULT"): type(t){}
  void start(string key){
    start_times[key] = std::chrono::high_resolution_clock::now();
  }
  void stop(string key){
    if (!start_times.count(key)){
      if (dynet::profiling_flag > 1)
        printf("[Timer]: %s not started\n", key.c_str());
      return;
    }
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_times[key]).count();
    if (type == "DEFAULT"){
      values[key] = elapsed;
    }
    else if (type == "ADD"){
      if (values.count(key) == 0)
          values[key] = 0.0;
        values[key] += elapsed;
    }
  }

  void show(){
    for (auto kv: values){
        printf("\t%s:\t%fms\n", kv.first.c_str(), kv.second);
    }
  }
  void clear(){
    values.clear();
    start_times.clear();
  }

  unordered_map<string, double> values;
  unordered_map<string, std::chrono::high_resolution_clock::time_point> start_times; 
};

class DynamicBatching {
public:
    vector<vector<int>> g;
    vector<vector<int>> g_r;
    vector<int> node2type;
    unordered_set<int> nodes;
    unordered_map<int, int> topo_value;
    unordered_map<int, int> type2weight;
    int node_id;
    int faketype_id;
    Timer localTimer;
    Timer addTimer;
    DynamicBatching(vector<vector<int> >& g_in, vector<int> &node2type_in, bool deleteUnBachable = false);
    void solve(std::vector<int>&best_batch_seq);
    // isFull: whether the subgraph is fully connected; used for incremental update of topo_sort
    void getContractSubgraphs(std::vector<std::unordered_set<int>> & subgraphs);
    void draw_graph(string filename, initializer_list<unordered_set<int> *> subgraphs, string graphName);
    void topo_sort();
    int contract(const std::vector<std::unordered_set<int> > & subgraphs);
    void graphHash(const unordered_set<int>&lower_nodes, const std::unordered_set<int>&upper_nodes, std::string& hash);
    void transitiveReduction(unordered_set<int>& redNodes);
    // min_batch, min_score
    enum search_t{
      ALL,
      LEAFPRUNE,
      ROOTPRUNE,
    };
    pair<int, int> bruteForce(unordered_set<int>& subgraph, vector<int> & opt_batch_seq, const search_t & s = ALL);
  
};
typedef dynet::nt::NodeType op_type;
// enum op_type{
//     lookup, logistic, gemm, add, act_tanh, cmult, concatenate
// };

extern unordered_map<int, string> type2name;
// void draw_graph(string filename, vector<vector<int> >& g, vector<int>& node2type, unordered_set<int> & nodes, string graphName = "G");
// void draw_graph(string filename, vector<vector<int> >& g, vector<int>& node2type);