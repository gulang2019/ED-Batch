#pragma once
#include "dynet/ooc-computation_graph.h"
#include <memory>

namespace OoC {
    
    // the symbolized expressions 
    class Expr;
    typedef std::shared_ptr<Expr> Expression;
    
    class Expr {
public:
        virtual void display()=0;
    };

    class Constant: public Expr{
public: 
        Constant(int v): value(v){}
        void display();
private:
        int value;
    };

    class Variable: public Expr {
public: 
        Variable(std::string name): name(name){}
        void display();
private:    
        std::string name;
    };

    class BinaryOp: public Expr {
public: 
        enum op_type{
            ADD,
            MULT,
            SUBSTRACT,
            EQ
        }type;

        BinaryOp(BinaryOp::op_type type, Expression lhs, Expression rhs): 
            type(type), lhs(lhs), rhs(rhs){}
        void display();
private:
        Expression lhs, rhs;
    };
        
    class IfThenElse: public Expr{
public:
        IfThenElse(Expression cond, Expression ifstmt, Expression elsestmt):
            cond(cond), ifstmt(ifstmt), elsestmt(elsestmt){}
        void display();
private:
        Expression cond, ifstmt, elsestmt;
    };

    class Array: public Expr {
public:
        Array(std::initializer_list<Expression> exprs): exprs(exprs){}
        void display();
private:
        std::vector<Expression> exprs;
    };
    
//     class Function: public Expr{
// public:
//         void display();
// private:
//         Function(Block & block): block(block){}
//         Block & block;
//     };

//     class Array: public Expr{
// public:
//         void display();
// private:
//         std::vector<Expression> elems;
//     };

    

    // frontend 
    Expression constant(int v);
    Expression variable(std::string name);
    Expression operator+(Expression e1, Expression e2);
    Expression if_then_else(Expression cond, Expression ifstmt, Expression elsestmt);
    Expression tuple(std::initializer_list<Expression> exprs);
    // Expression operator-(Expression e1, Expression e2);
    // Expression operator*(Expression e1, Expression e2);
    // Expression operator==(Expression e1, Expression e2);
    // Expression operator<(Expression e1, Expression e2);
    // Expression operator>(Expression e1, Expression e2);
    // the loopest
    class PerfectLoopNest:public Block {
public:
        void set_loop(
            Expression dim, 
            std::function<Expression(std::vector<Expression>)> dependency, 
            Expression loop_body);
private:
        Expression dim; // Array<Expr>
        Expression dependency; // Array<>
        Expression loop_body; // 
    };

} // namespace OoC