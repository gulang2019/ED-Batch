#include "dynet/ooc-mem.h"

using namespace std;

namespace OoC
{
    float *ConstView::operator[](vector<int> indices){
        assert(indices.size() == dims.size());
        int offset = 0;
        int i = 0;
        for (auto idx : indices)
            offset = offset * dims[i++] + idx;
        return v + offset*d.size();
    }

    float* RaggedView::operator[](vector<int> __indices){
        vector<int> indices(__indices);
        assert(indices.size() == 2);
        return v + (offsets[indices[0]] + indices[1])*d.size();
    }

    void RaggedView::transpose(bool reverse){
        sort(dims.begin(), dims.end());
        offsets.clear();
        offsets.resize(dims.back()+1);
        cout << "dim: ";
        for (auto dim:dims) cout << dim << ",";
        cout << endl;
        int last = 0;
        offsets[0] = 0;
        for (int i = 0; i < dims.size(); i++)
        {
            while (last != dims[i])
                offsets[++last] = dims.size() - i;
        }

        cout << "after offsets:";
        for (auto off: offsets)
            cout << off<<",";
        cout << endl;

        for (int i = 1; i < offsets.size(); ++i){
            offsets[i] += offsets[i - 1];
        }

        if (reverse) {
            cout << "before reverse:";
            for (auto off: offsets) 
                cout << off<<",";
            cout << endl;
            vector<int> __offsets(offsets.size());
            for (int i = 0; i < (int)offsets.size(); ++i){
                __offsets[i] = offsets.back() - offsets[i];
            }
            for (int i = 0, j = offsets.size()-1; i < (int)offsets.size(); ++i, --j){
                offsets[i] = __offsets[j];
            }
            cout << "after reverse:";
            for (auto off: offsets) 
                cout << off<<",";
            cout << endl;
        }
    }
} // namespace OoC