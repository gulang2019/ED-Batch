#include "dynet/ooc-computation_graph.h"
#include "dynet/lstm.h"

using namespace dynet;
using namespace OoC;
using namespace std;

namespace dynet
{

    void VanillaLSTMBuilder::define_bb()
    {
        SuperNode *bb_affine = new SuperNode(
            _cg,
            [&](const vector<Expression> &inputs, const vector<int> &params, vector<Expression> &outputs)
            {
                // assert(params.size() == )
                int i, has_prev_state, ;
                const vector<Expression>& vars = param_vars[i];
                Expression tmp;
                if (ln_lstm)
                {
                    const vector<Expression> &ln_vars = ln_param_vars[i];
                    if (has_prev_state)
                        tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]) + layer_norm(vars[_H2I] * i_h_tm1, ln_vars[LN_GH], ln_vars[LN_BH]);
                    else
                        tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]);
                }
                else
                {
                    if (has_prev_state)
                        tmp = affine_transform({vars[_BI], vars[_X2I], in, vars[_H2I], i_h_tm1});
                    else
                        tmp = affine_transform({vars[_BI], vars[_X2I], in});
                }
            },
            "VanillaLSTM::affine");
    }

} // namespace dynet