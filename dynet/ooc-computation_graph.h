#include "dynet.h"
#include "dynet/expr.h"

namespace OoC
{
    class SuperNode
    {
    public:
        typedef std::function<void(const std::vector<dynet::Expression> & input, const std::vector<int> & params, std::vector<dynet::Expression> & output)> func_t;
        SuperNode(dynet::ComputationGraph* cg, func_t f, std::string name="S"): _cg(cg), _func(f), _name(name){}
        std::vector<dynet::Expression> operator()(const std::vector<dynet::Expression> & input, const std::vector<int> & params, bool mark_basic_block = false);

    private:
        bool first_time = true;
        func_t _func;
        dynet::VariableIndex begin, end;
        dynet::ComputationGraph *_cg;
        std::unordered_map<dynet::VariableIndex, int> nid2aid;
        std::vector<int> output_indices;
        std::string _name;
        int _stid;
    };

    class FakeNode : public dynet::Node
    {
    public:
        FakeNode(dynet::Node *node) : dynet::Node(*node), _node(node) {}

        dynet::Dim dim_forward(const std::vector<dynet::Dim> &xs) const
        {
            return _node->dim_forward(xs);
        }
        std::string as_string(const std::vector<std::string> &args) const
        {
            return _node->as_string(args);
        }

        std::string as_dummy_string() const
        {
            return _node->as_dummy_string();
        }

        size_t aux_storage_size() const
        {
            return _node->aux_storage_size();
        }

        void forward_impl(const std::vector<const dynet::Tensor *> &xs,
                                  dynet::Tensor &fx) const
        {
            _node->forward_impl(xs, fx);
        }

        void backward_impl(const std::vector<const dynet::Tensor *> &xs,
                                   const dynet::Tensor &fx, const dynet::Tensor &dEdf, unsigned i,
                                   dynet::Tensor &dEdxi) const
        {
            _node->backward_impl(xs, fx, dEdf, i, dEdxi);
        }

        bool supports_multibatch() const { return _node->supports_multibatch(); }

        bool supports_multidevice() const { return _node->supports_multidevice(); }

        int autobatch_sig(const dynet::ComputationGraph &cg, dynet::SigMap &sm) const
        {
            if (sig_cache < 0){
                sig_cache = _node->autobatch_sig(cg, sm);
            }
            return sig_cache;
        }

        std::vector<int> autobatch_concat(const dynet::ComputationGraph &cg) const
        {
            return move(_node->autobatch_concat(cg));
        }

        dynet::Node *autobatch_pseudo_node(
            const dynet::ComputationGraph &cg,
            const std::vector<dynet::VariableIndex> &batch_ids) const
        {
            return _node->autobatch_pseudo_node(cg, batch_ids);
        }

        void autobatch_reshape(const dynet::ComputationGraph &cg,
                                       const std::vector<dynet::VariableIndex> &batch_ids,
                                       const std::vector<int> &concat,
                                       std::vector<const dynet::Tensor *> &xs,
                                       dynet::Tensor &fx) const
        {
            return _node->autobatch_reshape(cg, batch_ids, concat, xs, fx);
        }

    private:
        dynet::Node *_node;
        mutable int sig_cache = -1;
    };
} // namespace OoC