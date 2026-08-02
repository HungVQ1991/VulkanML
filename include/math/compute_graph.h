#pragma once

#include <vector>
#include "compute_node.h"

class Compute_Graph 
{
private:
    std::vector<Compute_Node> nodes;

public:
    void addNode(const Compute_Node& node) { nodes.push_back(node); }
    const std::vector<Compute_Node>& getNodes() const { return nodes; }
    void clear() { nodes.clear(); }
};