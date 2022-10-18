#include "dynet/tensor-eigen.h"
#include "dynet/nodes-ooc-argmax.h"

#include "dynet/nodes-impl-macros.h"


#include "dynet/tensor.h"
#include "dynet/index-tensor.h"

#ifdef __CUDACC__
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif

using namespace std;

namespace dynet {

// ************* ArgmaxIndex *************

#ifndef __CUDACC__

string ArgmaxIndex::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << ("argmax_indices(") << arg_names[0] << ")_{" << dim << '}';
  return s.str();
}

Dim ArgmaxIndex::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in ArgmaxIndex");
  // For now only support 1 dim
  DYNET_ARG_CHECK(xs[0].nd == 1, "ArgmaxIndex only supports vectors for now, got dimension " << xs);
  DYNET_ARG_CHECK(d < xs[0].nd, "Cannot compute argmax along dimension " << dim << " for tensor of shape " << xs);
  return Dim({xs[0].bd});
}

#endif

template<class MyDevice>
void ArgmaxIndex::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::TensorMap<Eigen::Tensor<unsigned, 1>>(reinterpret_cast<unsigned*>(fx.v), xs[0]->d.bd)
    .device(*dev.edevice) = tb<1>(*xs[0]).argmax(d).cast<unsigned>();
}

template<class MyDevice>
void ArgmaxIndex::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  // If we're using the straight-through estimator: copy gradient
  DYNET_ASSERT(false, "not implemented!");
  // Otherwise no gradient!
}
DYNET_NODE_INST_DEV_IMPL(ArgmaxIndex)

}
