#include "dynet/ooc-loop.h"

using namespace std;

namespace OoC{
    void BinaryOp::display(){
        lhs->display();
        switch (type){
            case ADD:
                cout << "+";
                break;  
            default:
                cout << "unk";
        };
        rhs->display();
    }

    void Constant::display(){cout << value;}

    void Variable::display(){cout << name;}

    void IfThenElse::display(){
        cond->display();
        cout << "?";
        ifstmt->display();
        cout << ":";
        elsestmt->display();
    }

    void Array::display(){
        cout << "(";
        for (auto& v: exprs) {
            v -> display();
            cout << ",";
        }
        cout <<")";
    }

    // expressions 
    Expression constant(int v) 
        {return make_shared<Constant>(v);}
    Expression variable(std::string name)
        {return make_shared<Variable>(name);}
    Expression operator+(Expression e1, Expression e2) 
        {return make_shared<BinaryOp>(BinaryOp::ADD, e1, e2);}
    Expression if_then_else(Expression cond, Expression ifstmt, Expression elsestmt)
        {return make_shared<IfThenElse>(cond, ifstmt, elsestmt);}
    Expression tuple(std::initializer_list<Expression>(exprs)) 
        {return make_shared<Array>(exprs);}
} // namespace OoC