#pragma once
#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <iostream>
#include <assert.h>

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include <dynet/lstm.h>
#include <dynet/param-init.h>

#define MAX_VOCAB_SIZE 256

namespace OoCTest
{
    const int verbose = 1;

    class Model
    {
    public:
        virtual dynet::Expression build_graph(dynet::ComputationGraph &cg,
                                 int batch_size = 32) = 0;
        virtual void reset() {}
    };

    struct ModelConfig
    {
        std::string name;
        int batch_size;
    };

    class DataLoader
    {
    public:
        void add_model(std::string name, Model *model)
        {
            models[name] = model;
        }

        dynet::Expression build_graph(
            dynet::ComputationGraph &cg,
            std::initializer_list<ModelConfig> configs)
        {
            std::vector<dynet::Expression> exprs;
            for (auto &config : configs)
            {
                if (!models.count(config.name)){
                    std::cerr << config.name << " is not in {";
                    for (auto& kv: models) std::cerr << kv.first << ",";
                    std::cerr << "}" << std::endl;
                    throw std::runtime_error("bad param");
                }
                dynet::Expression expr = models[config.name]->build_graph(cg, config.batch_size);
                exprs.push_back(expr);
            }
            return dynet::sum(exprs);
        }

        void reset(){
            for (auto kv: models) kv.second->reset();
        }

        std::unordered_map<std::string, Model *> models;
    };

} // namespace OoC