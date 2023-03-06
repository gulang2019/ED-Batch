#include <iostream>
#include <set>
#include "pqnode.h"
#include "pqtree.h"
#include "ext-pqtree.h"
#include <fstream>

using namespace std;


void read_file(const string& filename, vector<vector<vector<int> > >& pattern){
    ifstream file;
    file.open(filename);
    int n_node;
    file >> n_node;
    
    for (int i = 0; i < n_node; i++){

    }
}

void test(string filename) {
    
}

int main(){
    string filename = "./block/bilstm/G1.txt";
    test(filename);
    return 0;
}