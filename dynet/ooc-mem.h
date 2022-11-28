#ifndef OOC_MEM_H
#define OOC_MEM_H
#include "dynet/dim.h"
#include "dynet/globals.h"
#include "dynet/devices.h"

namespace OoC{
    struct View {
        View(const dynet::Dim& d):d(d){}
        void init() {
            assert(v == nullptr);
            v = (float*)dynet::default_device->pools[(int)dynet::DeviceMempool::FXS]->allocate(d.size() * size * sizeof(float));
        }
        virtual float* operator[] (std::vector<int>) = 0; 

        dynet::Dim d;
        int size;
        float* v = nullptr;
    };

    struct ConstView: public View{
        std::vector<int> dims;
        ConstView(const dynet::Dim& d, const std::vector<int> dims):
            View(d),dims(dims){
                size = 1;
                for (auto dim: dims) size *= dim;
            }
        float * operator[](std::vector<int>);
    };

    struct RaggedView:public View{
        std::vector<int> offsets;
        std::vector<int> dims;
        RaggedView(const dynet::Dim& d, const std::vector<int>dims):
        View(d), dims(dims){
            size = 0;
            offsets.resize(dims.size());
            int i = 0;
            for (auto len: dims) {
                offsets[i++] = size;
                size += len;
            }
        }
        float* operator[] (std::vector<int>) override;
        void transpose(bool reverse = false);
    };

} // namespace OoC

#endif