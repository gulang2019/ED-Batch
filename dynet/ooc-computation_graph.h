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
        SuperNode(func_t f, const std::string& name): 
            _func(f), _name(name){}
        std::vector<dynet::Expression> operator()(
            const std::vector<dynet::Expression> & input, 
            const std::vector<int> & const_params, 
            const std::vector<int> & runtime_params,
            bool mark_basic_block = false);
        void register_lookup(dynet::LookupParameter* lookup_param, int nid, int param_id){
            lookup_args.push_back({lookup_param, nid, param_id});
        }
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
        std::string _name;
        struct lookup_arg_t{
            dynet::LookupParameter* p;
            int nid;
            int param_id;
        };
        std::vector<lookup_arg_t> lookup_args;        
    };
} // namespace OoC