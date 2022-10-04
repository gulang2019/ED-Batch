#include "dynet.h"
#include "dynet/expr.h"
#include "utils.h"

namespace OoC
{
    class SuperNode
    {
    public:
        typedef std::function<void(const std::vector<dynet::Expression> & input, const std::vector<int> & params, std::vector<dynet::Expression> & output)> func_t;
        SuperNode(dynet::ComputationGraph* cg, func_t f, std::string name="S"): _cg(cg), _func(f), _name(name){}
        std::vector<dynet::Expression> operator()(const std::vector<dynet::Expression> & input, const std::vector<int> & params, bool mark_basic_block = false);
        static int reset(){
            int ret = n_node;
            n_node = 0;
            return ret;
        }
        static void synchronize();
        static int n_sync;

    private:
        static int n_node;
        static ThreadPool pool;
        static std::vector<std::future<int> > results;
        struct log_t{
            int first_time = true;
            dynet::VariableIndex begin, end;
            std::unordered_map<dynet::VariableIndex, int> nid2aid;
            std::vector<int> output_indices;
            int _stid;
        };
        TupleDict<log_t> params_dict; 
        func_t _func;
        dynet::ComputationGraph *_cg;
        std::string _name;
    };

    class FakeNode : public dynet::Node
    {
    public:
        FakeNode(dynet::Node *node, int nid = 0) : dynet::Node(*node), _node(node), _nid(nid) {}

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

        ~FakeNode(){
            fprintf(stdout, "FakeNode %d detructed\n", _nid);
        }

    private:
        dynet::Node *_node;
        mutable int sig_cache = -1;
        int _nid;
    };
} // namespace OoC