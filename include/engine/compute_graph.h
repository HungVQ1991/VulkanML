#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "compute_node.h"
#include "helper/magic_enum.hpp"

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

    void print()
    {
        Logger::logMessage(Input_Format{"Node count: {} nodes", nodes.size()}, Log_Level::LOG_INFO, true, 1, Log_Feature::GRAPH);
        for (const Compute_Node &node : nodes)
        {
            Logger::logMessage(Input_Format{"{}", static_cast<std::string>(magic_enum::enum_name<Compute_Pipeline>(node.pipeline_id))}, Log_Level::LOG_INFO, true, nodes.size(), Log_Feature::GRAPH);
        }
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