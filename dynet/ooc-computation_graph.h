#include "dynet.h"
#include "dynet/expr.h"
#include "utils.h"
#include "BS_thread_pool.hpp"

namespace OoC
{
    class SuperNode
    {
    public:
        typedef std::function<void(
            const std::vector<dynet::Expression> & input, 
            const std::vector<int> & const_params,
            const std::vector<int> & runtime_params, 
            std::vector<dynet::Expression> & output)> func_t;
        typedef std::function<void(
            const std::vector<dynet::Node*> nodes, 
            const std::vector<int> & params)> extra_func_t;
        SuperNode(func_t f, const std::string& name, extra_func_t extra_func = extra_func_t()): 
            _func(f), _name(name), _extra_func(extra_func){}
        std::vector<dynet::Expression> operator()(
            const std::vector<dynet::Expression> & input, 
            const std::vector<int> & const_params, 
            const std::vector<int> & runtime_params,
            bool mark_basic_block = false);
        static int new_graph(dynet::ComputationGraph *cg){
            assert(results.empty());
            _cg = cg;
            n_node = 0;
            n_sync = 0;
            return 0;
        }
        static void synchronize(std::string info = "");
        static int n_sync;
        static int n_node;

    private:
        static dynet::ComputationGraph *_cg;
        static BS::thread_pool pool;
        static std::vector<std::future<void*> > results;
        struct log_t{
            int first_time = true;
            dynet::VariableIndex begin, end;
            std::unordered_map<dynet::VariableIndex, int> nid2aid;
            std::vector<int> output_indices;
            int _stid;
        };
        TupleDict<log_t> params_dict; 
        func_t _func;
        extra_func_t _extra_func;
        std::string _name;
    };
} // namespace OoC