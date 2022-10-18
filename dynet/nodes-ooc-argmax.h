#ifndef DYNET_NODES_OOC_ARGMAX_H_
#define DYNET_NODES_OOC_ARGMAX_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y_i = 1 if i = argmax(x) else 0
struct ArgmaxIndex : public Node {
  explicit ArgmaxIndex(const std::initializer_list<VariableIndex>& a, unsigned d) : Node(a), d(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override {Sig s(nt::argmax_index); return sm.get_idx(s);} 
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override {return std::vector<int>(1,1);}
  virtual bool supports_multibatch() const override { return true; }
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                        const std::vector<VariableIndex> & batch_ids,
                                        const std::vector<int> & concat,
                                        std::vector<const Tensor*>& xs,
                                        Tensor& fx) const override {
        autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
    }
  unsigned d;
};

} // namespace dynet

#endif
