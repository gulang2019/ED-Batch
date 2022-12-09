#ifndef DYNET_EXEC_H
#define DYNET_EXEC_H

#include "dynet/dynet.h"

namespace dynet {

class DeviceManager;

class ExecutionEngine {
 public:
  virtual ~ExecutionEngine();
  virtual void invalidate() = 0;
  virtual void invalidate(unsigned) = 0;
  virtual const Tensor& forward() = 0;
  virtual const Tensor& forward(VariableIndex i) = 0;
  // forward on multiple nodes
  virtual std::vector<const Tensor*> forward(
      const std::vector<VariableIndex>& node_list);
  // if you want to add nodes and evaluate just the new parts
  virtual const Tensor& incremental_forward() = 0;
  virtual const Tensor& incremental_forward(VariableIndex i) = 0;
  virtual const Tensor& get_value(VariableIndex i) = 0;
  virtual const Tensor& get_gradient(VariableIndex i) = 0;
  virtual void backward(bool full = false) = 0;
  virtual void backward(VariableIndex i, bool full = false) = 0;
  virtual OoC::View* allocate_ragged(const Dim& d, const std::vector<int>& seqs, bool transpose = false, bool reverse = false){return nullptr;}
  virtual OoC::View* allocate(const Dim& d, const std::vector<int>& dims){return nullptr;}
  virtual void bind(int nid, OoC::View* view, const std::vector<int>& indices) {}
protected:
  explicit ExecutionEngine(ComputationGraph& cg);
  DeviceManager* const device_manager;
  ComputationGraph& cg;
  VariableIndex backward_computed;
};

class SimpleExecutionEngine : public ExecutionEngine {
 public:
  explicit SimpleExecutionEngine(ComputationGraph& cg) :
    ExecutionEngine(cg), num_nodes_evaluated(0) {}
  void invalidate() override;
  void invalidate(unsigned i) override;
  const Tensor& forward() override;
  const Tensor& forward(VariableIndex i) override;
  const Tensor& incremental_forward() override;
  const Tensor& incremental_forward(VariableIndex i) override;
  const Tensor& get_value(VariableIndex i) override;
  const Tensor& get_gradient(VariableIndex i) override;
  void backward(bool full = false) override;
  void backward(VariableIndex from_where, bool full = false) override;
 private:
  std::vector<Tensor> nfxs;
  std::vector<Tensor> ndEdfs;
  VariableIndex num_nodes_evaluated;
};

struct BatchInfo {
public:
  BatchInfo() : pseudo_node(nullptr) { }
  // The forward tensor, may be null if singleton batch
  Tensor nfx;
  // The pseudo node used for calculation, also may be null if not needed
  Node* pseudo_node;
  // IDs of the batch components
  std::vector<VariableIndex> ids;
  // 0=no need to concat
  // 1=need to concat
  // 2=need to concat + already contiguous in space
  std::vector<int> concat;
  // Concatenated arguments
  std::vector<const Tensor*> arg_nfxs;
  // The super node dim 
  int dim;

  bool mem_combine = false;
};

class BatchedExecutionEngine : public ExecutionEngine {
 public:
  explicit BatchedExecutionEngine(ComputationGraph& cg) :
    ExecutionEngine(cg), num_nodes_evaluated(0), num_batches_evaluated(0) {}
  ~BatchedExecutionEngine() { garbage_collect(); }
  void invalidate() override;
  void invalidate(unsigned i) override;
  const Tensor& forward() override;
  const Tensor& forward(VariableIndex i) override;
  const Tensor& incremental_forward() override;
  const Tensor& incremental_forward(VariableIndex i) override;
  const Tensor& get_value(VariableIndex i) override;
  const Tensor& get_gradient(VariableIndex i) override;
  
  // hack method for user to do schedule
  OoC::View* allocate_ragged(const Dim& d, const std::vector<int>&seqs, bool transpose = false, bool reverse = false);
  OoC::View* allocate(const Dim& d, const std::vector<int>& dims);
  void bind(int nid, OoC::View* view, const std::vector<int>&indices);
  void set_batches(const std::vector<std::vector<VariableIndex> >& batches);
  
  void backward(bool full = false) override;
  void backward(VariableIndex from_where, bool full = false) override;
  void garbage_collect();
  void visualize(int upto, std::string filename, std::string graphname, std::unordered_set<std::pair<int, int>, OoC::hash_pair> * mem_transfer_edges);
  void visualize_snode(int upto, std::string filname, std::string graphname, std::unordered_set<std::pair<int, int>, OoC::hash_pair> *mem_transfer_edges = nullptr);
  void export_graph(VariableIndex upto, std::string filename);
  void export_snode_graph(std::string filename);

 private:
  struct bind_t{
    int nid;
    OoC::View* view;
    std::vector<int> indices;
  };
  std::vector<OoC::View*> buffers;
  std::vector<bind_t> bindings;
  std::vector<std::vector<VariableIndex> > user_given_batches;

  static int graph_id;
  static OoC::DynamicBatching db;
  // static OoC::PatternCache pattern_cache;
  // static OoC::Trie head;
  // static std::vector<OoC::typeInfo> stypes;
  // static OoC::Scheduler& scheduler;
  // static std::vector<OoC::typeInfo> stypes;
  // a sophisticated implementation of OoC's inference stage
  void getBatches_typewiseLB(VariableIndex upto, VariableIndex& batch_id);
  void getBatches_rl(VariableIndex upto, VariableIndex& batch_id);
  void getBatches(VariableIndex upto, VariableIndex & batch_id);
  OoC::Timer localTimer;
  // std::vector<OoC::supernodeInfo> snodes;
  void construct_snode_graph_OoC(VariableIndex upto);
  
  enum {
    TRAIN,
    INFERENCE
  } schedule_mode;
  void schedule_snode_graph(std::string type);
  //allocate memory based on PQTree
  void pre_malloc(VariableIndex batch_id);
  void memory_allocation(BatchInfo & my_batch);
  void execute_batch(BatchInfo& batch);
  void execution(int upto);
  bool commit_batch_OoC(std::vector<int>& batch);
  void forward_OoC(VariableIndex upto);
  /**
   * autobatch_strategy: 
   *  1:  dynet
   *  2:  tf-fold
   *  8:  typewise-lb
   *  9:  rl 
   *  10: user defined batch configuration
   *  11: typewise-lb + rl
   *  12: dynet + do pre allocate
   */ 
  const Tensor& incremental_forward_no_update(VariableIndex upto,
                                              int autobatch_strategy);
  void combine_tensors(const std::vector<VariableIndex>& batch_ids,
                       int aid, Tensor &tout);
  void scatter_tensors(const std::vector<VariableIndex>& batch_ids, const Tensor &tin);
  void accumulate_tensors(const Tensor& tin,
                          const std::vector<VariableIndex>& batch_ids,
                          int ai);
  const Tensor& get_nfx(VariableIndex i);
  std::vector<Tensor> nfx_cache;
  std::vector<Tensor> ndEdfs;
  VariableIndex num_nodes_evaluated, num_batches_evaluated, num_batch_committed;
  // data structure for reversed scheduling
  std::vector<int> start_indices;
  // Information about the batched computation graph
  std::vector<VariableIndex> node2batch; // length: number of nodes
  std::vector<size_t> node2offset, node2size; // length: number of nodes
  std::vector<BatchInfo> batches; // length: number of batches
  std::vector<int> memory_affinity;
  int memory_affinity_tag = 0;
  SigMap sigmap;

  // For debug 
  std::vector<int> node2mem_pos;
  // the lower bound on batch numbers;
  int mem_id; 
  // whether the node is binded to user allocated memory
  std::vector<bool> pre_allocated;
  // pair<nid, offset>
  std::vector<int> pre_allocate_nodes;
  void pre_allocate_input(const std::vector<VariableIndex> & batch);
  inline void allocate_pre_allocated_input();
};

} // namespace dynet

#endif
