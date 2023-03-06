// extension of pqtree to support C2
// order-preserve constraints

#include "ext-pqtree.h"
#include <unordered_set>

using namespace std;

const int profiling_flag = 0;

set<int> Visitor::ExtractCanonicalSubsets(PQNode *node, set<set<int>> &canonical_subsets)
{
    // string str;
    // node->Print(&str);
    // fprintf(stdout, "ExtractCanonicalSubsets: %s, %d\n", str.c_str(), node->pertinent_child_count);
    set<int> ret;
    if (node->type_ == PQNode::leaf)
    {
        ret.insert(node->leaf_value_);
    }
    else if (node->type_ == PQNode::pnode)
    {
        for (auto child : node->circular_link_)
        {
            set<int> child_set = ExtractCanonicalSubsets(child, canonical_subsets);
            ret.insert(child_set.begin(), child_set.end());
        }
        canonical_subsets.insert(ret);
    }
    else if (node->type_ == PQNode::qnode)
    {
        PQNode *last = nullptr;
        PQNode *current = node->endmost_children_[0];
        set<int> neighbor_sets;
        while (current)
        {
            set<int> child_set = ExtractCanonicalSubsets(current, canonical_subsets);
            neighbor_sets.insert(child_set.begin(), child_set.end());
            if (neighbor_sets.size() > 1)
                canonical_subsets.insert(neighbor_sets);
            neighbor_sets = child_set;
            PQNode *next = current->QNextChild(last);
            last = current;
            current = next;
        }
    }
    return ret;
}

bool Visitor::InTree(PQNode *node, const PQNode *target)
{
    if (node == target)
        return true;
    if (node->type_ == PQNode::leaf)
    {
    }
    else if (node->type_ == PQNode::pnode)
    {
        for (auto child : node->circular_link_)
            if (InTree(child, target))
                return true;
    }
    else if (node->type_ == PQNode::qnode)
    {
        PQNode *last = nullptr;
        PQNode *current = node->endmost_children_[0];
        while (current)
        {
            if (InTree(current, target))
                return true;
            PQNode *next = current->QNextChild(last);
            last = current;
            current = next;
        }
    }
    return false;
}

void Visitor::ReplaceBinaryPNodes(PQNode *node)
{
    if (node->type_ == PQNode::leaf)
        return;
    if (node->type_ == PQNode::pnode)
    {
        for (auto child : node->circular_link_)
            ReplaceBinaryPNodes(child);
        if (node->circular_link_.size() == 2)
        {
            node->type_ = PQNode::qnode;
            node->endmost_children_[0] = node->circular_link_.front();
            node->endmost_children_[1] = node->circular_link_.back();
            node->circular_link_.front()->ClearImmediateSiblings();
            node->circular_link_.front()->immediate_siblings_[0] = node->circular_link_.back();
            node->circular_link_.back()->ClearImmediateSiblings();
            node->circular_link_.back()->immediate_siblings_[0] = node->circular_link_.front();
        }
    }
    else if (node->type_ == PQNode::qnode)
    {
        PQNode *last = nullptr;
        PQNode *current = node->endmost_children_[0];
        while (current)
        {
            ReplaceBinaryPNodes(current);
            PQNode *next = current->QNextChild(last);
            last = current;
            current = next;
        }
    }
    return;
}

void IsoPQTree::PQTree2Collection(list<set<int>> &collections)
{
    set<set<int>> collection_sets;
    Visitor::ExtractCanonicalSubsets(tree->Root(), collection_sets);
    for (auto &collection : collection_sets)
        collections.push_back(move(collection));
}

bool IsoPQTree::Reduce(const set<int> &reduction_set)
{
    return tree->Reduce(reduction_set);
}

bool IsoPQTree::ReduceAll(const set<set<int> > &reduction_sets)
{
    return  tree->ReduceAll(reduction_sets);
}

bool IsoPQTree::ReduceAll(const list<set<int> > &reduction_sets)
{
    return  tree->ReduceAll(reduction_sets);
}

bool IsoPQTree::ReduceAll(const vector<set<int> > &reduction_sets)
{
    return  tree->ReduceAll(reduction_sets);
}

string IsoPQTree::Print()
{
    string str;
    Print(tree->Root(), &str);
    return move(str);
}

bool IsoPQTree::IsoReduce(
    Pattern &pattern, PatternInfo &pattern_info, bool &has_update)
{
    set<set<int>> old_consecutive_cons;
    for (auto &subset : pattern_info.consecutive_constraints)
        old_consecutive_cons.insert(subset);

    set<set<int> > new_consecutive_constraints;
    for (auto &subset : pattern.subsets){
        bool succ = GetPatternInfo(set<int>(subset.begin(), subset.end()), new_consecutive_constraints);
        if(!succ) cout << "WARNING: get pattern info failed";
    }

    if (profiling_flag > 0)
    {
        fprintf(stdout, "[IsoReduce]: tree %s\n", Print().c_str());
        fprintf(stdout, "constriants:\n");
        for (auto &subset : new_consecutive_constraints)
        {
            fprintf(stdout, "{");
            for (auto ele : subset)
                fprintf(stdout, "%d, ", ele);
            fprintf(stdout, "}, \n");
        }
        fprintf(stdout, "\n");
    }
    pattern.transform(new_consecutive_constraints, pattern_info.consecutive_constraints);

    bool updated = (old_consecutive_cons != pattern_info.consecutive_constraints);

    if (profiling_flag > 0)
    {
        fprintf(stdout, "[IsoReduce::pattern]: ");
        pattern.show();
        fprintf(stdout, "\n");
        pattern_info.show();
        fprintf(stdout, "\n");
    }
    set<set<int>> consecutive_constraints;
    pattern.reverseTransform(pattern_info.consecutive_constraints, consecutive_constraints);
    if (profiling_flag > 0)
    {
        fprintf(stdout, "[IsoReduce %d]: ", updated);
        pattern.show();
        fprintf(stdout, "\n");
        fprintf(stdout, "\ttree:%s\n", Print().c_str());
        int idx = 0;
        for (auto &subset : consecutive_constraints)
        {
            fprintf(stdout, "\tC%d: (", idx++);
            for (auto ele : subset)
                fprintf(stdout, "%d, ", ele);
            fprintf(stdout, ")\n");
        }
        fprintf(stdout, "\n");
    }

    if (updated)
        has_update = true;
    else
        return true;
    if (profiling_flag > 0)
        tree->Root()->CheckReset();
    for (auto & subset: consecutive_constraints) {
        // fprintf(stdout, "\ttree: %s \n", tree->Print().c_str());
        if (!tree->SafeReduce(subset)) {
            pattern.valid = false;
            has_update = false;
            fprintf(stderr, "IsoReduce: conflict detected\n");
            return false;
        }
    }
    // if (!tree->ReduceAll(consecutive_constraints))
    // {
    //     fprintf(stderr, "IsoReduce: conflict detected\n");
    //     assert(false);
    //     return false;
    // }
    return true;
}

bool IsoPQTree::IsoReduceAll(
    vector<Pattern> &iso_patterns)
{
    int invalid_cnt = 0;
    bool suc;
    if (profiling_flag > 0)
    {
        fprintf(stdout, "[IsoReduceAll::patterns]: \n");
        for (auto &pattern : iso_patterns)
        {
            fprintf(stdout, "\t");
            pattern.show();
            fprintf(stdout, "\n");
        }
    }
    vector<PatternInfo> pattern_infos(iso_patterns.size());
    bool has_update = true;
    while (has_update)
    {
        has_update = false;
        int pid = 0;
        for (auto &pattern : iso_patterns)
        {
            if (pattern.valid) {
                suc = IsoReduce(pattern, pattern_infos[pid], has_update);
                invalid_cnt += suc == false;
            }
            pid++;
        }
    }


    Visitor::ReplaceBinaryPNodes(tree->Root());

    if (profiling_flag)
        fprintf(stdout, "[IsoReduceAll]: tree after transitive: %s\n", Print().c_str());

    for (auto &pattern : iso_patterns)
    {
        if (pattern.valid){
            suc = GetEquivalentClass(pattern);
            invalid_cnt += suc == false;
        }
    }
    if (profiling_flag)
        cout << "invalid cnt is " << invalid_cnt << endl; 

    if (profiling_flag > 0)
    {
        for (auto &qnodes : qnodes2dir)
        {
            fprintf(stdout, "equivalent nodes: ");
            for (auto &kv : qnodes)
            {
                string str;
                Print(kv.first, &str);
                fprintf(stdout, "(%s:%d), ", str.c_str(), kv.second);
            }
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "fixed pnodes: ");
        for (auto pnode : fixed_pnodes)
        {
            string str;
            Print(pnode, &str);
            fprintf(stdout, "%s, ", str.c_str());
        }
        fprintf(stdout, "\n");
    }

    if (!DecideQNodeDir())
        return false;

    if (profiling_flag > 0)
    {
        fprintf(stdout, "QNode's order after deciding direction:");
        for (auto &kv : qnode2dir)
        {
            string str;
            Print(kv.first, &str);
            fprintf(stdout, "(%s:%d), ", str.c_str(), kv.second);
        }
        fprintf(stdout, "\n");
    }

    return true;
}

bool IsoPQTree::IsoReduceAll(
    const vector<vector<vector<int>>> &iso_patterns)
{
    if (profiling_flag)
    {
        cout << "[IsoReduceAll::all patterns]:" << endl;
        for (auto& iso_pattern: iso_patterns){
            cout << "{";
            for (auto& subset: iso_pattern) {
                cout << "{";
                for (auto& ele: subset) 
                    cout << ele << ", ";
                cout << "}, ";
            }
            cout << "}," << endl;
        }
    }

    // TODO: check problem here;
    vector<Pattern> patterns;

    // firstly we reduce the output
    for (auto &iso_pattern: iso_patterns) {
        bool succ = tree->Reduce(set<int>(iso_pattern.front().begin(), iso_pattern.front().end()));
        assert(succ);
    }

    // the input should be exclusive and different from each other;
    for (auto &iso_pattern : iso_patterns)
    {
        unordered_set<int> S;
        int cnt = 0;
        int batch_size = iso_pattern.front().size();
        if (batch_size == 1) continue;
        vector<vector<int> > pattern;

        pattern.push_back(iso_pattern[0]);

        for (int idx = 1; idx < iso_pattern.size(); idx++)
        {
            auto & subset = iso_pattern[idx];
            set<int> tmp_S(subset.begin(), subset.end());
            if (tmp_S.size() == 1) continue;
            else if (tmp_S.size() != batch_size){
                {
                    fprintf(stdout, "[WARNING::IsoReduce]:require element in input-output pattern to be exclusive.\n");
                    fprintf(stdout, "{");
                    for (auto & subset: iso_pattern){
                        fprintf(stdout, "{");
                        for (auto ele: subset)
                            fprintf(stdout, "%d, ", ele);
                        fprintf(stdout, "}, ");
                    }
                    fprintf(stdout, "}\n");
                }
                continue;
            }
            else if (!tree->SafeReduce(tmp_S)){ // need opt
                if (profiling_flag){
                    cout << "Cannot reduce("; 
                    for (auto ele: tmp_S) cout << ele << ", ";
                    cout << ") on";
                    cout << Print() << endl; 
                }
                continue;
            }
            S.insert(subset.begin(), subset.end());
            cnt += subset.size();
            pattern.push_back(subset);
        }

        if ((int)S.size() < cnt)
        {
            if(profiling_flag > 0){
                fprintf(stdout, "[ERROR::IsoReduce]:require element in input-output pattern to be exclusive.\n");
                fprintf(stdout, "{");
                for (auto & subset: iso_pattern){
                    fprintf(stdout, "{");
                    for (auto ele: subset)
                        fprintf(stdout, "%d, ", ele);
                    fprintf(stdout, "}, ");
                }
                fprintf(stdout, "}\n");
            }
            assert(false);
            return false;
        }
        if (pattern.size() > 1) patterns.push_back({move(pattern)});
    }
    if (!patterns.size()) return true;

    return IsoReduceAll(patterns);
}

list<int> IsoPQTree::Frontier()
{
    list<int> ret;
    FindFrontier(tree->Root(), ret);
    return move(ret);
}

void IsoPQTree::FindFrontier(PQNode *node, list<int> &ordering)
{
    if (node->type_ == node->leaf)
    {
        ordering.push_back(node->leaf_value_);
    }
    else if (node->type_ == PQNode::pnode)
    {
        for (list<PQNode *>::iterator i = node->circular_link_.begin();
             i != node->circular_link_.end(); i++)
            FindFrontier(*i, ordering);
    }
    else if (node->type_ == PQNode::qnode)
    {
        PQNode *last = NULL;
        PQNode *current = qnode2dir.count(node) && qnode2dir.at(node) ? node->endmost_children_[0] : node->endmost_children_[1];
        while (current)
        {
            FindFrontier(current, ordering);
            PQNode *next = current->QNextChild(last);
            last = current;
            current = next;
        }
    }
}

void IsoPQTree::Print(PQNode *node, string *out) const
{
    if (node->type_ == PQNode::leaf)
    {
        char value_str[10];
        sprintf(value_str, "%d", node->leaf_value_);
        *out += value_str;
    }
    else if (node->type_ == PQNode::pnode)
    {
        *out += fixed_pnodes.count(node) ? "<" : "{";
        for (list<PQNode *>::const_iterator i = node->circular_link_.begin();
             i != node->circular_link_.end(); i++)
        {
            Print(*i, out);
            // Add a space if there are more elements remaining.
            if (++i != node->circular_link_.end())
                *out += " ";
            --i;
        }
        *out += fixed_pnodes.count(node) ? ">" : "}";
    }
    else if (node->type_ == PQNode::qnode)
    {
        *out += qnode2dir.count(node) ? "(" : "[";
        PQNode *last = NULL;
        PQNode *current = qnode2dir.count(node) && qnode2dir.at(node) ? node->endmost_children_[0] : node->endmost_children_[1];

        while (current)
        {
            Print(current, out);
            PQNode *next = current->QNextChild(last);
            last = current;
            current = next;
            if (current)
                *out += " ";
        }
        *out += qnode2dir.count(node) ? ")" : "]";
    }
}


bool IsoPQTree::Serialize(
    const vector<int>& subset, 
    vector<PQNode*>& nodes,
    unordered_map<PQNode*, set<int> >& node2leaf){
    if (!tree->Bubble(set<int>(subset.begin(), subset.end()), false)){
        return false;
    }
    
    queue<PQNode*> q;

    for (auto nid: subset){
        PQNode* node = tree->leaf_address_[nid];
        node->pertinent_leaf_count = 1;
        q.push(node);
        node2leaf[node] = {nid}; 
    }

    while(!q.empty()){
        PQNode* node = q.front();
        q.pop();
        if (node->type_ != PQNode::leaf) nodes.push_back(node);
        if (node->pertinent_leaf_count < (int) subset.size()){
            PQNode* parent = node->parent_;
            node2leaf[parent].insert(node2leaf[node].begin(), node2leaf[node].end());
            parent->pertinent_leaf_count += node->pertinent_leaf_count;
            parent->pertinent_child_count--;
            if (parent->pertinent_child_count == 0) 
                q.push(parent);
        }
    }

    tree->Reset();

    return true;
    
}

bool IsoPQTree::GetEquivalentClass(
    Pattern &pattern)
{

    if (profiling_flag > 0){
        fprintf(stdout, "[getEqaClass]: %s\n", Print().c_str());
        pattern.show();
        fprintf(stdout, "\n");
    }
    vector<vector<PQNode*> > serialized;
    vector<unordered_map<PQNode*, set<int> > > node2leaves;
    for (auto &subset: pattern.subsets){
        serialized.push_back({});
        node2leaves.push_back({});
        assert(Serialize(subset, serialized.back(), node2leaves.back()));
        
    }

    if (profiling_flag > 0)
    {
        fprintf(stdout, "serialized: \n");
        for (int i = 0; i < serialized.size(); i++){
            fprintf(stdout, "%d\t:", i);
            for (auto &node: serialized[i]) {
                string str;
                node->Print(&str);
                fprintf(stdout, "%s,", str.c_str());
            }
            fprintf(stdout, "\n");
        }
    }
    
    for (auto& vec: serialized) assert(vec.size() == serialized.front().size());

    bool suc = true;

    for (int i = 0; i < (int)serialized.front().size(); i++){
        vector<PQNode*> nodes;
        for (auto &vec: serialized) nodes.push_back(vec[i]);

        if (nodes.front()->type_ == PQNode::qnode){
            for (auto node: nodes) assert(node->type_ == PQNode::qnode);
            if (profiling_flag > 0){
                fprintf(stdout, "[EqaClass]:\n");
                for (int j = 0; j < nodes.size(); j++){
                    auto node = nodes[j];
                    string str;
                    node->Print(&str);
                    fprintf(stdout, "\t%s\n", str.c_str());    
                }
            }
            qnodes2dir.push_back({});
            
            map<set<int>, int> leaves2idx;
            size_t idx = 0;
            for (QNodeChildrenIterator it(nodes.front()); !it.IsDone(); it.Next()){
                auto node = it.Current();
                if (node2leaves[0].count(node)){
                    leaves2idx[pattern.transform(node2leaves[0][node])] = idx++;
                }
            }

            if (profiling_flag > 0){
                fprintf(stdout, "[leaves2idx] %d: \n", idx);
                for (auto & kv: leaves2idx) {
                    for (auto v: kv.first) fprintf(stdout, "%d, ", v);
                    fprintf(stdout, ": %d\n", kv.second);
                }
            }

            for (int j = 0; j < (int)nodes.size(); ++j){
                auto node = nodes[j];
                vector<int> indices;
                for (QNodeChildrenIterator it(node); !it.IsDone(); it.Next()){
                    auto node = it.Current();
                    if (node2leaves[j].count(node)){
                        indices.push_back(leaves2idx[pattern.transform(node2leaves[j][node])]);
                    }
                }
                if (indices.size() != idx){
                    string str;
                    node->Print(&str);
                    fprintf(stdout, "indices %d %d %s: ", idx, j, str.c_str());
                    for(auto i: indices) fprintf(stdout, "%d, ", i);
                    fprintf(stdout, "\n");
                }
                assert(indices.size() == idx);

                if (idx > 1){
                    int gap = indices[1] - indices[0];
                    suc &= (gap == 1 || gap == -1);
                    for (int k = 1; k < indices.size(); k++)
                        suc &= (indices[k] - indices[k - 1]) == gap;
                }

                if (!suc) {
                    pattern.valid = false;
                    return false;
                }

                bool dir = indices[0] == 0;
                if (qnodes2dir.back().count(node))
                    if (dir != qnodes2dir.back()[node]) {
                        if (profiling_flag){
                            pattern.show();
                            fprintf(stdout, "\n");
                            string str;
                            node->Print(&str);
                            fprintf(stdout, "node: %s\n", str.c_str());
                        }
                        continue;
                    }
                qnodes2dir.back()[node] = dir;
            }
        }
        else if (nodes.front()->type_ == PQNode::pnode){
            for (auto node: nodes) assert(node->type_ == PQNode::pnode);
            
            int nid = 0;

            for (int j = 0; j < (int)nodes.size(); ++j)
            {
                if (fixed_pnodes.count(nodes[j]))
                {
                    nid = j;
                    break;
                }
            }
            // Todo: seems to be buggy here why no conflict for PNode? 
            map<set<int>, int> leaves2idx;
            int idx = 0;
            for (auto child : nodes[nid]->circular_link_)
                leaves2idx[pattern.transform(node2leaves[nid][child])] = idx++;

            for (int j = 0; j < (int)nodes.size(); ++j)
            {
                nodes[j]->circular_link_.sort([&leaves2idx, &node2leaves, &pattern, j](PQNode *child1, PQNode *child2)
                                                    { return leaves2idx[pattern.transform(node2leaves[j][child1])] <
                                                             leaves2idx[pattern.transform(node2leaves[j][child2])]; });
            }
            fixed_pnodes.insert(nodes.begin(), nodes.end());
        }
    }

    return true;
}

bool IsoPQTree::DecideQNodeDir()
{
    vector<bool> fixed(qnodes2dir.size(), false);
    int invalid_cnt = 0;
    for (int i = 0; i < (int)qnodes2dir.size(); i++)
    {
        if (fixed[i])
            continue;
        queue<int> Q;
        Q.push(i);
        while (!Q.empty())
        {
            int qid = Q.front();
            Q.pop();
            if (fixed[qid])
                continue;
            fixed[qid] = true;
            bool suc = true;
            for (auto kv: qnodes2dir[qid]){
                if (qnode2dir.count(kv.first) && qnode2dir[kv.first] != kv.second){
                    suc = false;
                    invalid_cnt ++;
                    break;
                }
            }
            if (!suc) continue;
            for (auto kv : qnodes2dir[qid])
                qnode2dir[kv.first] = kv.second;

            for (int j = 0; j < (int)qnodes2dir.size(); j++)
            {
                if (!fixed[j])
                {
                    auto this_it = qnodes2dir[qid].begin();
                    auto that_it = qnodes2dir[j].begin();
                    int reverse_cnt = 0, intersect_cnt = 0;
                    // the intersect 
                    while (this_it != qnodes2dir[qid].end() &&
                           that_it != qnodes2dir[j].end())
                    {
                        if (this_it->first < that_it->first)
                            this_it++;
                        else if (this_it->first > that_it->first)
                            that_it++;
                        else
                        {
                            intersect_cnt++;
                            reverse_cnt += this_it->second != that_it->second;
                            this_it++;
                            that_it++;
                        }
                    }
                    if (intersect_cnt > 0)
                    {
                        if (intersect_cnt == reverse_cnt || reverse_cnt == 0)
                            Q.push(j);
                        if (intersect_cnt == reverse_cnt) {
                            for (auto &kv : qnodes2dir[j])
                                kv.second = !kv.second;
                        }
                        else if (reverse_cnt != 0) {
                            fixed[j] = true;
                            invalid_cnt += 1;
                        }
                    }
                }
            }
        }
    }
    if (invalid_cnt) {
        fprintf(stdout, "[DecideQNodeDir::Warning]: %d/%ld invalid\n", invalid_cnt, qnodes2dir.size());
    }
    return true;
}

bool IsoPQTree::GetPatternInfo(const set<int> &subset, set<set<int>> &consecutive_constraints)
{
    if (!tree->Bubble(subset, true)){
        tree->Reset();
        tree->CleanPseudo();
        return false;
    }

    if (tree->pseudonode_)
    {
        Visitor::ExtractCanonicalSubsets(tree->pseudonode_, consecutive_constraints);
        tree->Reset();
        tree->CleanPseudo();
        return true;
    }

    queue<PQNode *> q;
    for (auto ele : subset)
    {
        PQNode *node = tree->leaf_address_[ele];
        assert(node);
        node->pertinent_leaf_count = 1;
        q.push(node);
    }

    while (!q.empty())
    {
        PQNode *node = q.front();
        q.pop();
        if (node->pertinent_leaf_count < (int)subset.size())
        {
            PQNode *parent = node->parent_;
            parent->pertinent_leaf_count +=
                node->pertinent_leaf_count;
            parent->pertinent_child_count--;
            if (parent->pertinent_child_count == 0)
                q.push(parent);
        }
        else
        {
            if (node->type_ == PQNode::pnode)
            {
                Visitor::ExtractCanonicalSubsets(node, consecutive_constraints);
            }
            else if (node->type_ == PQNode::qnode)
            {
                PQNode *last = nullptr;
                PQNode *current = node->endmost_children_[0];
                set<int> neighbor_consecutive_constraints;
                while (current)
                {
                    if (current->pertinent_leaf_count)
                    {
                        set<int> this_child_leaves = Visitor::ExtractCanonicalSubsets(current, consecutive_constraints);
                        bool is_begin = neighbor_consecutive_constraints.size() == 0;
                        neighbor_consecutive_constraints.insert(this_child_leaves.begin(), this_child_leaves.end());
                        if (!is_begin)
                            consecutive_constraints.insert(neighbor_consecutive_constraints);
                        neighbor_consecutive_constraints = this_child_leaves;
                    }
                    PQNode *next = current->QNextChild(last);
                    last = current;
                    current = next;
                }
            }
            assert(q.empty());
        }
    }
    tree->Reset();
    return true;
}


bool IsoPQTree::Check(const vector<vector<vector<int> > >& patterns, bool verbose){
    auto frontiers = Frontier();

    if (verbose) {
        cout << "test frontier: ";
        for (auto node: frontiers) cout << node << ", ";
        cout << endl;
        cout << "tree: " << Print() << endl;
        cout << "patterns:" << endl;
        for (auto & pattern: patterns){
            cout << "{";
            for (auto & subset: pattern) {
                cout << "(";
                for (auto ele: subset) cout << ele << ", ";
                cout << "),";
            }
            cout << "}" << endl;
        }
    }


    vector<int> mem_pos(frontiers.size(), -1);
    int idx = 0;
    for (auto v: frontiers){
        mem_pos[v] = idx++;
    }
    int invalid_cnt = 0;
    for (auto& pattern: patterns){
        auto & examplar = pattern[0];
        vector<int> positions;
        for (auto v: examplar) positions.push_back(mem_pos[v]);
        sort(positions.begin(), positions.end());
        int tmp = positions.front();
        bool invalid = false;
        for (auto v: positions){
            if (v != tmp++) {
                if (verbose){
                    cerr << "examplar:";
                    for (auto v: examplar) cerr << v << ",";
                    cerr << "mem_pos:";
                    for (auto v: examplar) cerr << mem_pos[v] << ",";
                    cerr << " check failed!" << endl;
                }
                invalid = true;}
        }
        for (auto & subset: pattern){
            if (subset.size() != examplar.size()) {
                if (verbose){
                    cerr << "subset size check failed!" << endl;
                }
                invalid = true;}
            for (int i = 0; i < (int)subset.size(); ++i){
                if (mem_pos[subset[i]] < 0) {
                    if (verbose){
                        cerr << "mem_pos check failed!" << endl;
                    }
                    invalid = true;
                }
                if ((mem_pos[examplar[i]] - mem_pos[examplar[0]]) != (mem_pos[subset[i]] - mem_pos[subset[0]])){
                    if (verbose){
                     cerr << "align check failed!" << endl;
                    }
                    invalid = true;
                }
            }
        }
        invalid_cnt += invalid;
    }
    if (invalid_cnt) 
        cout << invalid_cnt << " of " << patterns.size() << "is invalid." << endl;
    else cout << "Check passed!" << endl;
    return true;
}
/*
1. subsets --> roots;
2. >=3 PNode:
    recognize by an equivalent class data structure
3. QNode && ==2 PNode:
    create an pos neg vector for each pattern;
solve the pos neg satisfiability problem;
prepare the result.
*/

/*
we must be able to know whether or not to
Divide QNode into partitions
*/