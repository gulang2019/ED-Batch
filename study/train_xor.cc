#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include <iostream>

using namespace std;
using namespace dynet;


int main(int argc, char ** argv){
    dynet::initialize(argc, argv);

    const unsigned ITERATIONS = 30;

    ParameterCollection m;
    SimpleSGDTrainer trainer(m);

    const unsigned HIDDEN_SIZE = 8;
    Parameter p_W = m.add_parameters({HIDDEN_SIZE, 2});
    Parameter p_b = m.add_parameters({HIDDEN_SIZE});
    Parameter p_V = m.add_parameters({1, HIDDEN_SIZE});
    Parameter p_a = m.add_parameters({1});

    ComputationGraph cg;
    Expression W = parameter(cg, p_W);
    Expression b = parameter(cg, p_b);
    Expression V = parameter(cg, p_V);
    Expression a = parameter(cg, p_a);

    vector<dynet::real> x_values(2);
    Expression x = input(cg, {2}, &x_values);
    dynet::real y_value;
    Expression y = input(cg, &y_value);

    Expression h = tanh(W * x + b);
    Expression y_pred = V * h + a;
    Expression loss_expr = squared_distance(y_pred, y);

    cg.print_graphviz();

    for (unsigned iter = 0; iter < ITERATIONS; iter ++){
        double loss = 0.0;
        for (unsigned mi = 0; mi < 4; mi ++){
            bool x1 = mi % 2;
            bool x2 = (mi / 2) % 2;
            x_values[0] = x1? 1: -1;
            x_values[1] = x2? 1: -1;
            y_value = (x1 ^ x2)? -1: 1;
            loss += as_scalar(cg.forward(loss_expr));
            cg.backward(loss_expr);
            trainer.update();
        }
        loss /= 4;
        cerr << "E = " << loss <<endl;
    }

    vector<pair<int, int> > vec ({{-1,1}, {-1,-1}, {1,-1},{1,1}});
    for (auto p: vec){
        x_values[0] = p.first;
        x_values[1] = p.second;
        cg.forward(y_pred);
        printf("[%d %d] %d: %f \n", p.first, p.second, p.first * p.second, as_scalar(y_pred.value()));
    }

    TextFileSaver saver("xor.model");
    saver.save(m);
    
    return 0;
}