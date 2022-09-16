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
 protected:
  explicit ExecutionEngine(const ComputationGraph& cg);
  DeviceManager* const device_manager;
  const ComputationGraph& cg;
  VariableIndex backward_computed;
};

class SimpleExecutionEngine : public ExecutionEngine {
 public:
  explicit SimpleExecutionEngine(const ComputationGraph& cg) :
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
};

class BatchedExecutionEngine : public ExecutionEngine {
 public:
  explicit BatchedExecutionEngine(const ComputationGraph& cg) :
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
  void backward(bool full = false) override;
  void backward(VariableIndex from_where, bool full = false) override;
  void garbage_collect();
  void visualize(int upto, std::string filename, std::string graphname, std::unordered_set<std::pair<int, int>, OoC::hash_pair> * mem_transfer_edges);
  void visualize_snode(std::string filname, std::string graphname);
  void export_graph(VariableIndex upto, std::string filename);
  void export_snode_graph(std::string filename);
 private:
  
  static int graph_id;
  static OoC::DynamicBatching db;
  static OoC::PatternCache pattern_cache;
  static OoC::Trie head;
  static std::vector<OoC::typeInfo> stypes;
  static OoC::Scheduler& scheduler;
  // static std::vector<OoC::typeInfo> stypes;
  // a sophisticated implementation of OoC's inference stage
  void getBatches(VariableIndex upto, VariableIndex & batch_id);
  OoC::Timer localTimer;
  std::vector<OoC::supernodeInfo> snodes;
  void construct_snode_graph_OoC(VariableIndex upto);
  void construct_snode_graph_from_bb_OoC(VariableIndex upto);
  void schedule_snode_graph_OoC();
  
  enum {
    TRAIN,
    INFERENCE
  } schedule_mode;
  void schedule_snode_graph_rl();
  // store execution order and do memory allocation
  void memory_allocation(BatchInfo & my_batch);
  void execution(int upto);
  bool commit_batch_OoC(int tid);
  void forward_OoC(VariableIndex upto);

  const Tensor& incremental_forward_no_update(VariableIndex upto,
                                              int autobatch_strategy);
  void combine_tensors(const std::vector<VariableIndex>& batch_ids,
                       int aid, Tensor &tout);
  void accumulate_tensors(const Tensor& tin,
                          const std::vector<VariableIndex>& batch_ids,
                          int ai);
  const Tensor& get_nfx(VariableIndex i);
  std::vector<Tensor> nfx_cache;
  std::vector<Tensor> ndEdfs;
  VariableIndex num_nodes_evaluated, num_batches_evaluated, num_batch_committed;
  // Information about the batched computation graph
  std::vector<VariableIndex> node2batch; // length: number of nodes
  std::vector<size_t> node2offset, node2size; // length: number of nodes
  std::vector<BatchInfo> batches; // length: number of batches
  static SigMap sigmap;

  // For debug 
  std::vector<int> node2sid;
  std::vector<int> node2mem_pos;
  int mem_id; 
  void visualize_trie();

};

} // namespace dynet

#endif
