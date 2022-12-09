#include "dynet/tensor-eigen.h"
#include "dynet/nodes-matrixmultiply.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/matrix-multiply.h"

using namespace std;

namespace dynet
{

  // ************* MatrixMultiply *************

#ifndef __CUDACC__

  string MatrixMultiply::as_string(const vector<string> &arg_names) const
  {
    ostringstream s;
    s << arg_names[0] << " * " << arg_names[1];
    return s.str();
  }

  Dim MatrixMultiply::dim_forward(const vector<Dim> &xs) const
  {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in MatrixMultiply")
    DYNET_ARG_CHECK(xs[0].cols() == xs[1].rows(), "Mismatched input dimensions in MatrixMultiply: " << xs);
    DYNET_ARG_CHECK(xs[0].nd <= 2 && xs[1].nd <= 2, "Cannot multiply tensors of dimension higher than 2: " << xs);
    if (xs[1].ndims() == 1)
      return Dim({xs[0].rows()}, max(xs[0].bd, xs[1].bd));
    return Dim({xs[0].rows(), xs[1].cols()}, max(xs[0].bd, xs[1].bd));
  }

  int MatrixMultiply::autobatch_sig(const ComputationGraph &cg, SigMap &sm) const
  {
    // Currently assumes there are two args, and batches with a shared first arg.
    // TODO do we want to treat different dimensions of first/second arg differently?
    // if(dim.bd == 1) {
    Sig s(nt::matmul);
    s.add_int(dim.bd);
    if (sharedA)
      s.add_node(args[0]);
    else
      s.add_dim(cg.nodes[args[0]]->dim);
    s.add_dim(cg.nodes[args[1]]->dim);
    return sm.get_idx(s);
    // } else {
    //   return 0; // TODO handle the batched case as well? should it differ at all?
    // }
  }

  std::vector<int> MatrixMultiply::autobatch_concat(const ComputationGraph &cg) const
  {
    vector<int> ret(args.size(), 0);
    if (dim.bd >= 1)
    {
      ret[1] = 1;
    }
    if (!sharedA)
      ret[0] = 1;
    return ret;
  }

  void MatrixMultiply::autobatch_reshape(const ComputationGraph &cg,
                                         const std::vector<VariableIndex> &batch_ids,
                                         const std::vector<int> &concat,
                                         std::vector<const Tensor *> &xs,
                                         Tensor &fx) const
  {
    if (sharedA)
    {
      autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
    }
    else
    {
      // fx[b, N, M] = A[b, M, K] * B[b, N, K];
      // fx[b, N, M] = B[b,N,K] @ A[b,M,K].T
      const Node *examplar = cg.nodes[batch_ids[0]];
      auto &dimA = cg.nodes[examplar->args[0]]->dim;
      auto &dimB = cg.nodes[examplar->args[1]]->dim;
      unsigned b = batch_ids.size();
      unsigned M = dimA.rows();
      unsigned K = dimA.cols();
      unsigned N = dimB.bd;
      const_cast<Tensor *>(xs[0])->d = Dim({{M, K}}, b);
      const_cast<Tensor *>(xs[1])->d = Dim({{N, K}}, b);
      fx.d = Dim({{N, M}}, b);
    }
  }

#endif

  template <class MyDevice>
  void MatrixMultiply::forward_dev_impl(const MyDevice &dev, const vector<const Tensor *> &xs, Tensor &fx) const
  {
    DYNET_ASSERT(xs.size() == 2, "Failed dimension check in MatrixMultiply::forward");
    DYNET_ARG_CHECK(fx.d.bd == max(xs[0]->d.bd, xs[1]->d.bd), "Failed dimension check in MatrixMultiply::forward");
    // fx = mat(fx0) + xs[0] * xs[1]
    // cout << "matmul: x0," << xs[0]->d << ", x1:" << xs[1]->d << endl;
    // xs[0]{1024,2048X3},xs[1]{1,2048X3},fx{1,1024X3}
    if (sharedA)
      dynet::MatrixMultiply(dev, *xs[0], *xs[1], fx, dev.kSCALAR_ZERO);
    else
      MatrixMultiplyTransp(dev, *xs[1], *xs[0], fx);
  }

  template <class MyDevice>
  void MatrixMultiply::backward_dev_impl(const MyDevice &dev,
                                         const vector<const Tensor *> &xs,
                                         const Tensor &fx,
                                         const Tensor &dEdf,
                                         unsigned i,
                                         Tensor &dEdxi) const
  {
    DYNET_ASSERT(i < 2, "Failed dimension check in MatrixMultiply::backward");
    // y = A * B
    if (i == 0)
    {
      // dA = dy * B^T
      MatrixMultiplyTranspAcc(dev, dEdf, *xs[1], dEdxi);
    }
    else
    {
      // dB = A^T * dy
      MatrixTranspMultiplyAcc(dev, *xs[0], dEdf, dEdxi);
    }
  }
  DYNET_NODE_INST_DEV_IMPL(MatrixMultiply)

}
