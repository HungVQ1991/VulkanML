#include "engine/compute_graph.h"

#include <string>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"

void Compute_Graph::addNode(const Compute_Node &_node)
{
    nodes.push_back(_node);
}

void Compute_Graph::addNode(Compute_Node &&_node)
{
    nodes.push_back(std::move(_node));
}

void Compute_Graph::clear() noexcept
{
    nodes.clear();
}

void Compute_Graph::print()
{
    Logger::logMessage(Input_Format{"Node count: {} nodes", nodes.size()}, Log_Level::LOG_INFO, true, 1, Log_Feature::GRAPH);
    for (const Compute_Node &node : nodes)
    {
        Logger::logMessage(Input_Format{"{}", static_cast<std::string>(magic_enum::enum_name<Compute_Pipeline>(node.pipeline_id))}, Log_Level::LOG_INFO, true, nodes.size(), Log_Feature::GRAPH);
    }
}
