#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "compute_node.h"

class Compute_Graph
{
private:
    std::vector<Compute_Node> nodes;

public:
    Compute_Graph() = default;
    ~Compute_Graph() = default;

    Compute_Graph(const Compute_Graph &) = default;
    Compute_Graph &operator=(const Compute_Graph &) = default;

    Compute_Graph(Compute_Graph &&other) noexcept = default;
    Compute_Graph &operator=(Compute_Graph &&other) noexcept = default;

    void addNode(const Compute_Node &_node)
    {
        nodes.push_back(_node);
    }

    void addNode(Compute_Node &&_node)
    {
        nodes.push_back(std::move(_node));
    }

    void clear() noexcept
    {
        nodes.clear();
    }

    const Compute_Node &getNode(size_t _index) const { return nodes.at(_index); }
    Compute_Node &getNode(size_t _index) { return nodes.at(_index); }
    const std::vector<Compute_Node> &getNodes() const noexcept { return nodes; }
    std::vector<Compute_Node> &getNodes() noexcept { return nodes; }
    size_t getNodeCount() const noexcept { return nodes.size(); }
    bool isEmpty() const noexcept { return nodes.empty(); }

    void setNodes(const std::vector<Compute_Node> &_nodes) { nodes = _nodes; }
    void setNodes(std::vector<Compute_Node> &&_nodes) noexcept { nodes = std::move(_nodes); }
};