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
#include <memory>
#include <queue>
#include <functional>
#include <stack>
#include <math.h>
#include <list>
#include <map>

#include "utils.h"
// #include "dynet/dynet.h"

namespace OoC
{

#define TO_OFFSET (8)
#define FROM_OFFSET (16)
#define LINEAR_OFFSET (24)
#define LINEAR_FLAG (0x12345678)
#define TO_MASK (0xff00)
#define FROM_MASK (0xff0000)
#define THIS_MASK (0xff)

  extern Timer ooc_timer;

  extern int profiling_flag;

  struct hash_pair
  {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2> &p) const
    {
      auto hash1 = std::hash<T1>{}(p.first);
      auto hash2 = std::hash<T2>{}(p.second);

      if (hash1 != hash2)
      {
        return hash1 ^ hash2;
      }

      // If hash1 == hash2, their XOR is zero.
      return hash1;
    }
  };

  class DynamicBatching;
  class Pattern;

  template<typename T>
  struct Trie
  {
    std::unordered_map<int, Trie *> next;
    bool isLeaf = false;
    T data;
  };

  struct supernodeInfo
  {
    std::vector<int> succs;
    int type;
    int inputCnt = 0;
    int min_nid;
    int bid;
  };

  struct BBInfo{
    int stid;
    int nop;
    std::vector<int> input_nodes;
  };
  
  struct typeInfo
  {
    int cnt = 0;
    std::vector<int> frontiers; // for scheduling
    Pattern *pattern;
    BBInfo * bbinfo;
    void show(){
      for (auto ele: frontiers){
        fprintf(stdout, "%d, ", ele);
      }
    }
  };


  class Pattern
  {
  public:
    int id;
    int n_batch;
    int nop;
    std::vector<int> batch_seq;
    std::unordered_map<int, int> distr;
    std::vector<int> nodes;
    std::vector<int> types;

    std::vector<std::vector<int>> batch_ids;
    std::vector<int> mem_allocation_order;
    void show();
  };

  class PatternCache
  {
  public:
    std::unordered_map<int, std::unique_ptr<Pattern>> patterns;
    std::unordered_set<std::pair<int, int>, hash_pair> boundary_edges;
    std::unordered_set<std::pair<int, int>, hash_pair> tr_edges;
    std::unordered_set<std::pair<int, int>, hash_pair> internal_edges;

    PatternCache(DynamicBatching *db_in) : db(db_in) {}
    PatternCache() {}
    double update(std::unordered_map<int, std::vector<int>> &type2batchSeqs);
    // return the score of the portion of node mapped
    double inference();
    double update_tr_edges(const std::vector<std::pair<int, int>> &new_tr_edges);
    void transitive_reduction(const std::unordered_set<int> &nodes);
    void update_internal_edges(const std::unordered_set<int> &subgraph);
    bool hash_check(std::unordered_set<std::pair<int, int>, hash_pair> &edges);
    bool hash_check(std::unordered_set<std::pair<int, int>, hash_pair> &edges, std::unordered_set<int> &subgraph);
    bool topo_check(int idx, const std::unordered_set<int> &subgraph);
    bool topo_check(int idx, std::vector<int> &subgraph);

    void get_batch_ids_dynet(const std::vector<std::vector<int>> &ops, const std::vector<int>& node2types, Pattern *pattern);
    void get_batch_ids_fold(const std::vector<std::vector<int>> &ops, const std::vector<int>& node2types, Pattern *pattern);
    void get_batch_ids_ooc(const std::vector<std::vector<int>> &ops, const std::vector<int>& node2types, Pattern *pattern);

    void get_batch_ids(const std::vector<std::vector<int>> &ops, std::vector<int>& node2types, Pattern *pattern, 
      std::string alg);
    Pattern* add_pattern(int key, const std::vector<std::vector<int>> &ops, const std::vector<int>& node2types, 
      std::string alg = "ooc");
    
    Pattern get_pattern(const std::vector<std::vector<int>> &ops, const std::vector<int>& node2types, 
      std::string alg = "ooc");

  private:
    DynamicBatching *db;
  };
  

  class Distribution
  {
  public:
    std::unordered_map<int, int> distribution;
    int sum;
    double delta_entropy(std::unordered_map<int, int> &updates, int factor = 1);
    void update(std::unordered_map<int, int> &updates, int factor = 1, bool setSum = false);
    double getEntropy();
    void show();
  };

  class DynamicBatching
  {
  public:
    // graph related data structure
    std::vector<std::vector<int>> g;
    std::vector<std::vector<int>> g_r;
    std::vector<std::vector<int>> old_g_r;
    std::unordered_map<int, int> topo_value;

    // type related data structures
    int faketype_id;
    std::vector<int> node2type;
    std::unordered_map<int, int> type2weight;
    std::unordered_map<int, std::vector<int>> type2BatchSeq;
    std::unordered_map<int, int> type2nop;
    std::unordered_map<int, int> node2nop;
    std::unordered_set<int> linearTypes;
    std::unique_ptr<Distribution> distribution_ptr;
    // type2its super type
    std::unordered_map<int, int> type2father;
    std::unordered_set<int> root_types;

    // node related data structure
    int node_id;
    int n_node_input;
    int n_unbatchable;
    std::unordered_set<int> nodes;
    bool isHashed;
    std::vector<int> node_hash;
    std::vector<int> node2father; // (train) the child-father tree

    Timer localTimer;
    Timer addTimer;

    std::unique_ptr<PatternCache> pattern_cache;
    std::unordered_map<std::pair<int, int>, std::unordered_set<int>, hash_pair> boundary_edges;

    enum mode_t
    {
      NOCACHE,
      TRAIN,
      INFERENCE
    } mode;
    double cache_hit_rate;
    const double train2inference_thres = 0.8;
    const double inference2train_thres = 0.8;
    int n_train_iteration;
    int n_inference_iteration;
    inline void update_mode();

    // mode_str = default | train;
    // score_func_str = tfidf | information_entropy
    DynamicBatching(std::string mode_str = "default", std::string score_func_str = "tfidf");
    DynamicBatching(std::vector<std::vector<int>> &g_in, std::vector<int> &node2type_in, bool deleteUnBachable = false);
    void setGraph(std::vector<std::vector<int>> &g_in, std::vector<int> &node2type_in, bool deleteUnBachable = false);

    // main functions
    void solve(std::vector<int> &best_batch_seq);
    // isFull: whether the subgraph is fully connected; used for incremental update of topo_sort
    enum score_func_t
    {
      TFIDF,
      INFORMATION_ENTROPY
    } score_func;
    bool findOptPatternAndContract(std::vector<std::unordered_set<int>> &contracted_subgraphs, double T = 1);
    std::vector<int> contract(std::vector<std::unordered_set<int>> &subgraphs, const std::vector<int> &typeBeginPoints, bool isLinear);
    int contract_update_type(std::vector<std::unordered_set<int>>::iterator begin, std::vector<std::unordered_set<int>>::iterator end, int base_op_id, bool isLinear);
    void contract_update_graph(std::vector<std::unordered_set<int>> &subgraphs, bool isLinear);
    int getDistribution(const std::unordered_set<int> &subgraph, std::unordered_map<int, int> &distr);
    // min_batch, min_score
    enum search_t
    {
      ALL,
      MEMORY,
      LEAFPRUNE,
      ROOTPRUNE,
    };
    std::pair<int, int> bruteForce(std::unordered_set<int> &subgraph, std::vector<int> &opt_batch_seq, const search_t &s = ALL, Pattern *pattern = nullptr);
    int mem_cost(const std::vector<std::vector<int>> &batches, const std::unordered_set<int> &subgraph, const int maxIdx, std::list<int> &memIds);
    // utils
    void draw_graph(std::string filename, std::initializer_list<std::unordered_set<int> *> subgraphs, std::string graphName);
    void draw_graph(std::string filename, std::vector<std::unordered_set<int> *> &subgraphs, std::string graphname, std::vector<int> *hashKeys = nullptr);
    void draw_boundary_edges();
    void graphHash(const std::unordered_set<int> &lower_nodes, const std::unordered_set<int> &upper_nodes, std::string &hash);
    void topo_sort(std::unordered_set<int> subgraph, std::vector<int> *topo_seq);
    void transitiveReduction(std::unordered_set<int> &redNodes, bool update_cache = false);
    int hashSubgraph(const std::unordered_set<int> &subgraph, int boundary_condition, std::unordered_set<std::pair<int, int>, hash_pair> *outEdges);
    void hashNodes();
    // hash nodes by dfs embedding of type and neighbor type
    std::pair<int, int> greedyTreeSCS(std::unordered_set<int> &subgraph, std::vector<int> &opt_batch_seq);
    void backPropogateNode2Father();
    void forwardType2Father(int newest_type);
    void recoverState(int minType);
  };

  enum op_type
  {
    unbatchable=0, tanh, sqrt, abs, erf, square, cube, exp, logsigmoid, loggamma, log, nobackprop, scalegradient, identity, negate, rectify, logistic, softsign, silu, round, ceiling, floor,
    sinh, cosh, asinh, acosh, atanh, sin, cos, tan, asin, acos, atan, plus_const, concat, cmult, csum, sum, squared_distance, softmax, pnls, pickrange, scalar_mult, dropout,
    input, scalar_input, lookup, select, argmax_index,
    COMPLEX,
    affine, matmul, transpose,
    vanilla_lstm_gates, vanilla_lstm_h, vanilla_lstm_c,
    conv2d, block, get, loss, END
  };

  // typedef dynet::nt::NodeType op_type;
  // enum op_type{
  //   unbatchable, logistic, lookup, gemm, add, act_tanh, cmult, concatenate, reduce_sum, loss, parameter
  // };

  extern std::unordered_map<int, std::string> type2name;
  struct Env
  {
    struct Info
    {
      double sum = 0;
      double square_sum = 0;
      int n_trial = 0;
    };
    std::vector<Info> infos;
    double get_reward(int t, int v)
    {
      if (t >= (int)infos.size())
      {
        infos.push_back({});
      }
      assert(t < infos.size());
      auto &info = infos[t];
      info.sum += (double)v;
      info.square_sum += (double)v * (double)v;
      info.n_trial++;
      double ex2 = info.square_sum / (double)info.n_trial;
      if (ex2 < 0){
        fprintf(stdout, "sum, square_sum, n_trial, ex2, t, v: %f, %f, %d, %lf, %d, %d\n",
          info.sum, info.square_sum, info.n_trial, ex2, t, v);
        assert(false);
      }
      double ex = info.sum / (double)info.n_trial;
      double edev = std::sqrt(ex2 - ex * ex) + 1e-6;
      double reward = -(v - ex) / edev;
      assert(!(reward!=reward));
      // if (reward!=reward){
      //   fprintf(stdout, "sum, square_sum, n_trial, ex2, ex, edev, t, v: %d, %d, %d, %f, %f, %f, %d, %d\n",
      //     info.sum, info.square_sum, info.n_trial, ex2, ex, edev, t, v);
      //   assert(false);
      // }
      return reward;
    }
    void show(){
      fprintf(stdout, "--------------env---------------\n");
      int t = 0;
      for (auto & info: infos){
        double ex2 = info.square_sum / (double)info.n_trial;
        double ex = info.sum / (double)info.n_trial;
        double edev = std::sqrt(ex2 - ex * ex) + 1e-6;
        fprintf(stdout, "%d: ave %f, dev %f, n_trial %d\n", 
          t++, ex, edev, info.n_trial);
      }
    }
    void reset()
    {
      infos.clear();
    }
  };

  class Scheduler
  {
    struct nodeInfo
    {
      std::vector<int> succs;
      int type;
      int inputCnt = 0;
    };
  public:
    int train(const std::vector<supernodeInfo*> &snodes, const std::vector<typeInfo> &stypes, int n_trial = 20);
    int train(const std::vector<nodeInfo>& __snodes, const std::unordered_map<int, typeInfo>& __stypes);
    void train(std::string filename, int n_trial = 1);
    virtual int train() {return 0;};
    virtual int batch_train(int batch_size = 15) {return 0;};
    virtual int get_action(const std::set<int> &state) {return *state.begin();};
    virtual ~Scheduler(){};
    void visualize(std::string filename);
    void validate(std::string filename);
    void show_frontier();
    void show();
    int lower_bound();
  
  private:
    void prepare_data_from_file(const std::string filename);

  protected:
    int commit(int tid);
    void reversed_commit(int tid, const std::vector<int> &snode_batch);
    void reset();

    
    std::vector<nodeInfo> _snodes;
    std::unordered_map<int, typeInfo> _stypes;

    std::vector<nodeInfo> snodes;
    std::unordered_map<int, typeInfo> stypes;
    Env env;
    int max_depth;
    int n_node;
    
    int n_remain;
    int n_batch;
    std::unordered_set<int> commited;

    int verbose = 0;
  };

  class QLearningModel: public Scheduler
  {
  public:
    QLearningModel(
        int n_train_iter = 2000,
        int n_step = 20,
        double alpha = 0.3,
        double _gamma = 0.95,
        double q_init = 1,
        bool require_rho = false
    ):  n_train_iter(n_train_iter), 
        n_step(n_step), 
        alpha(alpha), _gamma(_gamma), 
        q_init(q_init), require_rho(require_rho)
    {
      State::model = this;
    }

    int train() override;
    int get_action(const std::set<int> &state) override;
    int batch_train(int batch_size) override;

  private:
    int n_train_iter = 3000;
    int n_step = 20;
    double alpha = 0.3;
    double _gamma = 0.95;
    double q_init = 1;
    bool require_rho = false;

    int step(bool inference = true);
    

    struct State
    {
      static QLearningModel *model;
      std::unordered_map<int, double> Q;
      int take_action(bool inference = false);
      void show();
    };

    int n_state;
    TupleDict<State> states;
    std::unordered_map<State *, int> state2id;
    State *get_curr_state(bool inference = false);
    void show_states();
  };


  class ParallelQLearningModel: public Scheduler{
  public:
    ParallelQLearningModel(
        int n_train_iter = 2500,
        int n_step = 20,
        double alpha = 0.3,
        double _gamma = 0.95,
        double q_init = 1,
        bool require_rho = false
    ):  n_train_iter(n_train_iter), 
        n_step(n_step), 
        alpha(alpha), _gamma(_gamma), 
        q_init(q_init), require_rho(require_rho)
    {}

    inline int get_action(const std::set<int> &state) override;
    int batch_train(int batch_size = 10) override;

  private:
    int n_train_iter = 3000;
    int n_step = 20;
    double alpha = 0.3;
    double _gamma = 0.95;
    double q_init = 1;
    bool require_rho = false;
    std::unique_ptr<Scheduler> _model = nullptr;
  };


  /*usage: 
    vector<supernodeInfo*> snodes;
    StaticScheduler* s = new MyScheduler(snodes);
    vector<int> batch;
    while(!s->get_next_batch(batch)){
      f(batch);
    }
  */ 
  class StaticScheduler {
    public:
      StaticScheduler(std::vector<supernodeInfo*> & nodes, std::vector<typeInfo> & types): 
      _nodes(nodes), _types(types){
        show();
      }
      int lower_bound();
      void show();
      virtual bool get_next_batch(std::vector<int>&batch) = 0;
      virtual ~StaticScheduler(){}
    protected:
      std::vector<supernodeInfo*>& _nodes; 
      std::vector<typeInfo>& _types;
      int n_batch = 0;
  };

  class TypewiseLBScheduler: public StaticScheduler{
    struct Node {
      std::vector<int> succs;
      int inputCnt = 0;
    }; 
  public: 
    TypewiseLBScheduler(std::vector<supernodeInfo*>& _nodes, std::vector<typeInfo>& _types): 
      StaticScheduler(_nodes, _types){
      init();
    }
    bool get_next_batch(std::vector<int>&batch) override;
  private:
    int n_rand = 0;
    std::vector<Node> nodes;
    std::vector<int> frontier_cnt;
    void init();
  };

  class DynetScheduler: public StaticScheduler {
  public: 
    DynetScheduler(std::vector<supernodeInfo*>& _nodes, std::vector<typeInfo>& _types): 
      StaticScheduler(_nodes, _types){
        init();
    }
    void init();
    bool get_next_batch(std::vector<int>& batch) override;
  private: 
    std::vector<double> depth;
    std::vector<int> cnt;
  };

  class TFFoldScheduler: public StaticScheduler {
  public: 
    TFFoldScheduler(std::vector<supernodeInfo*>& _nodes, std::vector<typeInfo>& _types): 
      StaticScheduler(_nodes, _types){
        init();
    }
    void init();
    bool get_next_batch(std::vector<int>& batch) override;
  private: 
    std::vector<int> depth;
    std::vector<int> exec_order;
    size_t idx = 0;
  };

  class LearnableScheduler: public StaticScheduler{
  public:
    LearnableScheduler(std::vector<supernodeInfo*>& _nodes, std::vector<typeInfo>& _types):
      StaticScheduler(_nodes, _types){
        if (!trained){
          init();
          trained = true;
        }
    }
    bool get_next_batch(std::vector<int>& batch) override;
  private:
    static bool trained;
    static Scheduler* scheduler;
    void init();
  };
} // namespace OoC

/*
ALG:
DB.set (g, node2type)
DB.solve() {
  batchSeqs = []
  OnlineLearning(Cache, G)
  while(StopCondition(G)){
    pattern, subgraphs = Extractor::getSubgraph()
    Cache.insert(pattern, score)
    pattern, subgraphs = getContractSubgraphs()
    contract(subgraphs)
    batchSeq = bruteforce(pattern)
    batchSeqs.push_back(batchSeq)
    prunes()
  }
}

OnlineLearning(Cache, G){
  for i <- 1 to n do:
    w1,i = 1;
  for t <- 1 to T do:
    Receive(x_t)
    Receive(y_t)
    for i <- 1 to N do:
      w_{t+1, i} <- w_t_i
  experts = cache.patterns
  scores = []
  for expert: experts:
    scores.push_back(Score(G, expert))
}

solve:  5494.629708ms                                                               │(dynet) [gulang2020@scc build]$ make
  getContractSubgraphs:   2130.118726ms                                               │Scanning dependencies of target dynet
  bruteForce:     377.466342ms                                                        │[  1%] Building CXX object dynet/CMakeFiles/dynet.dir/exec.cc.o
  transitiveReduction:    2142.782679ms                                               │[  2%] Linking CXX shared library libdynet.so
  contract:       436.416555m

Extractor::getContractSubgraph(){
  1. classify edges by equivalent relationship and calculate scores
    No overlap; DFS
  2. classify equivalent classes by type topology (graph hasing); find the optimal pattern;
    Coarse-grained graph hasing --> detailed comparision
  3. make returns;
}

Matcher::getContractSubgraph(){
  1. patterns = cache.optimalPatterns
  for pattern in patterns:
    subgraphs = findMatches(G, pattern)
    return subgraphs
}

f(step, cache) {

}


*/