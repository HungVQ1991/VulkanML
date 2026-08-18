#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <format>
#include <algorithm>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "helper/logger.h"
#include "vulkan_context.h"
#include "vulkan_network.h"
#include "shader_generator.h"
#include "pipeline_cache_manager.h"
#include "shader_dictionary.h"

#ifndef ENABLE_EXECUTOR_DEBUG_LOGS
#define ENABLE_EXECUTOR_DEBUG_LOGS 0
#endif

#if ENABLE_EXECUTOR_DEBUG_LOGS
#define EXECUTOR_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define EXECUTOR_LOG_DEBUG(msg) ((void)0)
#endif

#ifndef ENABLE_FUSION_SHADOW_VALIDATION
#define ENABLE_FUSION_SHADOW_VALIDATION 0
#endif

#if ENABLE_FUSION_SHADOW_VALIDATION
#define SHADOW_SAMPLING_RATE 100
#define SHADOW_EPSILON 2e-4f
#endif

class Graph_Executor
{
private:
    const Vulkan_Context &context;
    const Vulkan_Network &network;
    Pipeline_Cache_Manager &cache_manager;
    const Shader_Dictionary &shader_dict;

    mutable std::vector<std::string> has_print_terminal;

    VkCommandBuffer cmd_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> descriptor_cache[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorPool descriptor_pools[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};

    static std::string replacePlaceholders(
        std::string text,
        const std::vector<std::string> &inputs,
        const std::vector<bool> &is_input_reg,
        const std::vector<std::string> &outputs,
        const std::vector<bool> &is_output_reg,
        std::uint32_t pc_word_offset)
    {
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            std::string token_prefix = std::format("{{in_{}}}[", i);
            std::string token_plain = std::format("{{in_{}}}", i);

            if (is_input_reg[i])
            {
                std::size_t pos = 0;
                while ((pos = text.find(token_prefix, pos)) != std::string::npos)
                {
                    std::size_t end_pos = text.find(']', pos);
                    if (end_pos != std::string::npos)
                    {
                        text.replace(pos, end_pos - pos + 1, inputs[i]);
                        pos += inputs[i].length();
                    }
                    else
                    {
                        break;
                    }
                }
            }

            std::size_t pos = 0;
            while ((pos = text.find(token_plain, pos)) != std::string::npos)
            {
                text.replace(pos, token_plain.length(), inputs[i]);
                pos += inputs[i].length();
            }
        }

        for (std::size_t i = 0; i < outputs.size(); ++i)
        {
            std::string token_prefix = std::format("{{out_{}}}[", i);
            std::string token_plain = std::format("{{out_{}}}", i);

            if (is_output_reg[i])
            {
                std::size_t pos = 0;
                while ((pos = text.find(token_prefix, pos)) != std::string::npos)
                {
                    std::size_t end_pos = text.find(']', pos);
                    if (end_pos != std::string::npos)
                    {
                        text.replace(pos, end_pos - pos + 1, outputs[i]);
                        pos += outputs[i].length();
                    }
                    else
                    {
                        break;
                    }
                }
            }

            std::size_t pos = 0;
            while ((pos = text.find(token_plain, pos)) != std::string::npos)
            {
                text.replace(pos, token_plain.length(), outputs[i]);
                pos += outputs[i].length();
            }
        }

        for (int i = 31; i >= 0; --i)
        {
            std::string token = std::format("{{pc_{}}}", i);
            std::size_t pos = 0;
            while ((pos = text.find(token, pos)) != std::string::npos)
            {
                std::string replacement = std::format("pc.data[{}]", pc_word_offset + i);
                text.replace(pos, token.length(), replacement);
                pos += replacement.length();
            }
        }
        return text;
    }

    std::vector<std::uint32_t> getExternalBufferIndices(const Compute_Node &node) const
    {
        std::unordered_map<std::uint32_t, std::size_t> last_write_op;
        for (std::size_t op_idx = 0; op_idx < node.fused_operations.size(); ++op_idx)
        {
            const auto &op = node.fused_operations[op_idx];
            for (std::uint32_t out_idx : op.output_buffer_indices)
            {
                last_write_op[out_idx] = op_idx;
            }
        }

        std::vector<std::uint32_t> external_indices;
        std::unordered_set<std::uint32_t> added_set;

        for (std::size_t op_idx = 0; op_idx < node.fused_operations.size(); ++op_idx)
        {
            const auto &op = node.fused_operations[op_idx];
            for (std::uint32_t in_idx : op.input_buffer_indices)
            {
                if (!last_write_op.contains(in_idx) && !added_set.contains(in_idx))
                {
                    external_indices.push_back(in_idx);
                    added_set.insert(in_idx);
                }
            }
        }

        for (std::uint32_t ext_idx : node.external_output_indices)
        {
            if (!added_set.contains(ext_idx))
            {
                external_indices.push_back(ext_idx);
                added_set.insert(ext_idx);
            }
        }

        for (const auto &op : node.fused_operations)
        {
            const auto &meta = shader_dict.getMetadata(op.pipeline_id);
            for (std::uint32_t persistent_out_local : meta.persistent_outputs)
            {
                if (persistent_out_local < op.output_buffer_indices.size())
                {
                    std::uint32_t persistent_buf_idx = op.output_buffer_indices[persistent_out_local];
                    if (!added_set.contains(persistent_buf_idx))
                    {
                        external_indices.push_back(persistent_buf_idx);
                        added_set.insert(persistent_buf_idx);
                    }
                }
            }
        }

        if (!node.fused_operations.empty())
        {
            for (std::uint32_t out_idx : node.fused_operations.back().output_buffer_indices)
            {
                if (!added_set.contains(out_idx))
                {
                    external_indices.push_back(out_idx);
                    added_set.insert(out_idx);
                }
            }
        }

        return external_indices;
    }

    std::string generateFusedGlsl(const Compute_Node &node) const
    {
        std::uint32_t local_x = 256;
        std::uint32_t local_y = 1;
        std::uint32_t local_z = 1;

        Op_Class primary_op_class = Op_Class::ELEMENTWISE;
        Compute_Pipeline primary_op = Compute_Pipeline::ADD;

        bool has_reduction = false;

        if (!node.fused_operations.empty())
        {
            primary_op = node.fused_operations[0].pipeline_id;
            primary_op_class = shader_dict.getMetadata(primary_op).op_class;

            if (primary_op_class == Op_Class::MATRIX_2D || primary_op_class == Op_Class::TENSOR_3D)
            {
                local_x = 16;
                local_y = 16;
                local_z = 1;
            }

            for (const auto &op : node.fused_operations)
            {
                if (shader_dict.getMetadata(op.pipeline_id).op_class == Op_Class::STANDALONE)
                {
                    has_reduction = true;
                    break;
                }
            }
        }

        Shader_Generator gen(local_x, local_y, local_z);

        if (has_reduction)
        {
            gen.enableSubgroupOperations();
        }

        std::vector<std::uint32_t> external_indices = getExternalBufferIndices(node);
        std::unordered_set<std::uint32_t> external_buffer_set(external_indices.begin(), external_indices.end());

        for (std::uint32_t buf_idx : external_indices)
        {
            gen.addBuffer(buf_idx, std::format("buf_{}", buf_idx));
        }

        gen.setPushConstants("uint data[32];");

        std::unordered_map<std::uint32_t, std::string> register_map;

        if (primary_op_class == Op_Class::MATRIX_2D)
        {
            gen.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
            gen.addLogicSnippet("    uint r = gl_GlobalInvocationID.y;");

            std::string bound_r = "pc.data[0]";
            std::string bound_c = "pc.data[2]";

            if (primary_op == Compute_Pipeline::LINEAR_BACKWARD_INPUT || primary_op == Compute_Pipeline::TRANSPOSE)
            {
                bound_r = "pc.data[0]";
                bound_c = "pc.data[1]";
            }
            else if (primary_op == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS)
            {
                bound_r = "pc.data[1]";
                bound_c = "pc.data[2]";
            }
            else if (primary_op == Compute_Pipeline::SOFTMAX || primary_op == Compute_Pipeline::SOFTMAX_BACKWARD)
            {
                bound_r = "pc.data[0]";
                bound_c = "";
            }

            if (!bound_c.empty())
            {
                gen.addLogicSnippet(std::format("    if (r < {} && c < {}) {{", bound_r, bound_c));
            }
            else
            {
                gen.addLogicSnippet(std::format("    if (r < {}) {{", bound_r));
            }
        }
        else if (primary_op_class == Op_Class::STANDALONE)
        {
            gen.addLogicSnippet("    uint r = gl_WorkGroupID.x;");
            gen.addLogicSnippet("    if (r < pc.data[0]) {");
        }
        else if (primary_op_class == Op_Class::TENSOR_3D)
        {
            if (primary_op == Compute_Pipeline::CONV2D_FORWARD_PASS)
            {
                gen.addLogicSnippet("    uint oc = gl_GlobalInvocationID.x;");
                gen.addLogicSnippet("    uint ow = gl_GlobalInvocationID.y;");
                gen.addLogicSnippet("    uint n_oh = gl_GlobalInvocationID.z;");
                gen.addLogicSnippet("    uint n = n_oh / pc.data[4];");
                gen.addLogicSnippet("    uint oh = n_oh % pc.data[4];");
                gen.addLogicSnippet("    if (oc < pc.data[6] && ow < pc.data[5] && oh < pc.data[4] && n < pc.data[0]) {");
            }
            else if (primary_op == Compute_Pipeline::MAXPOOL2D_FORWARD)
            {
                gen.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                gen.addLogicSnippet("    uint ow = gl_GlobalInvocationID.y;");
                gen.addLogicSnippet("    uint n_oh = gl_GlobalInvocationID.z;");
                gen.addLogicSnippet("    uint n = n_oh / pc.data[4];");
                gen.addLogicSnippet("    uint oh = n_oh % pc.data[4];");
                gen.addLogicSnippet("    if (c < pc.data[3] && ow < pc.data[5] && oh < pc.data[4] && n < pc.data[0]) {");
            }
            else if (primary_op == Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT)
            {
                gen.addLogicSnippet("    uint ic = gl_GlobalInvocationID.x;");
                gen.addLogicSnippet("    uint iw = gl_GlobalInvocationID.y;");
                gen.addLogicSnippet("    uint n_ih = gl_GlobalInvocationID.z;");
                gen.addLogicSnippet("    uint n = n_ih / pc.data[1];");
                gen.addLogicSnippet("    uint ih = n_ih % pc.data[1];");
                gen.addLogicSnippet("    if (ic < pc.data[3] && iw < pc.data[2] && ih < pc.data[1] && n < pc.data[0]) {");
            }
            else if (primary_op == Compute_Pipeline::MAXPOOL2D_BACKWARD)
            {
                gen.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                gen.addLogicSnippet("    uint iw = gl_GlobalInvocationID.y;");
                gen.addLogicSnippet("    uint n_ih = gl_GlobalInvocationID.z;");
                gen.addLogicSnippet("    uint n = n_ih / pc.data[1];");
                gen.addLogicSnippet("    uint ih = n_ih % pc.data[1];");
                gen.addLogicSnippet("    if (c < pc.data[3] && iw < pc.data[2] && ih < pc.data[1] && n < pc.data[0]) {");
            }
            else if (primary_op == Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD)
            {
                gen.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                gen.addLogicSnippet("    uint n = gl_GlobalInvocationID.y;");
                gen.addLogicSnippet("    if (c < pc.data[3] && n < pc.data[0]) {");
            }
            else if (primary_op == Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD)
            {
                gen.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                gen.addLogicSnippet("    uint iw = gl_GlobalInvocationID.y;");
                gen.addLogicSnippet("    uint n_ih = gl_GlobalInvocationID.z;");
                gen.addLogicSnippet("    uint n = n_ih / pc.data[1];");
                gen.addLogicSnippet("    uint ih = n_ih % pc.data[1];");
                gen.addLogicSnippet("    if (c < pc.data[3] && iw < pc.data[2] && ih < pc.data[1] && n < pc.data[0]) {");
            }
            else
            {
                gen.addLogicSnippet("    if (gl_GlobalInvocationID.x < pc.data[0]) {");
            }
        }
        else
        {
            gen.addLogicSnippet("    if (global_id < pc.data[0]) {");
        }

        for (std::size_t op_idx = 0; op_idx < node.fused_operations.size(); ++op_idx)
        {
            const Fused_Operation &op = node.fused_operations[op_idx];
            const Snippet_Metadata &meta = shader_dict.getMetadata(op.pipeline_id);

            if (meta.shared_mem_size > 0)
            {
                gen.enableSubgroupOperations();
                gen.addSharedMemory(meta.shared_mem_size, "shared_mem");
            }

            std::vector<std::string> inputs;
            std::vector<bool> is_input_reg;

            for (std::uint32_t in_idx : op.input_buffer_indices)
            {
                if (register_map.contains(in_idx))
                {
                    inputs.push_back(register_map[in_idx]);
                    is_input_reg.push_back(true);
                }
                else
                {
                    inputs.push_back(std::format("buf_{}", in_idx));
                    is_input_reg.push_back(false);
                }
            }

            std::vector<std::string> outputs;
            std::vector<bool> is_output_reg;
            std::vector<std::string> post_statements;

            for (std::size_t local_out_i = 0; local_out_i < op.output_buffer_indices.size(); ++local_out_i)
            {
                std::uint32_t out_idx = op.output_buffer_indices[local_out_i];
                bool is_accumulator = std::find(meta.accumulator_output_indices.begin(),
                                                meta.accumulator_output_indices.end(),
                                                static_cast<std::uint32_t>(local_out_i)) != meta.accumulator_output_indices.end();
                bool is_persistent = std::find(meta.persistent_outputs.begin(),
                                               meta.persistent_outputs.end(),
                                               static_cast<std::uint32_t>(local_out_i)) != meta.persistent_outputs.end();

                if (is_accumulator || is_persistent)
                {
                    outputs.push_back(std::format("buf_{}", out_idx));
                    is_output_reg.push_back(false);
                }
                else
                {
                    std::string reg_name = gen.getUniqueVar("reg");
                    gen.addLogicSnippet(std::format("        float {} = 0.0;", reg_name));

                    register_map[out_idx] = reg_name;
                    outputs.push_back(reg_name);
                    is_output_reg.push_back(true);

                    if (external_buffer_set.contains(out_idx))
                    {
                        post_statements.push_back(std::format("        buf_{}[global_id] = {};", out_idx, reg_name));
                    }
                }
            }

            std::uint32_t pc_word_offset = op.pc_offset / 4;
            std::string snippet = replacePlaceholders(meta.glsl_template, inputs, is_input_reg, outputs, is_output_reg, pc_word_offset);

            auto remove_str = [](std::string &s, const std::string &to_remove)
            {
                std::size_t pos = 0;
                while ((pos = s.find(to_remove, pos)) != std::string::npos)
                {
                    s.erase(pos, to_remove.length());
                }
            };

            if (op_idx == 0)
            {
                if (primary_op_class == Op_Class::MATRIX_2D)
                {
                    remove_str(snippet, "uint r = gl_GlobalInvocationID.y;");
                    remove_str(snippet, "uint c = gl_GlobalInvocationID.x;");
                }
                else if (primary_op_class == Op_Class::TENSOR_3D)
                {
                    remove_str(snippet, "uint oc = gl_GlobalInvocationID.x;");
                    remove_str(snippet, "uint ic = gl_GlobalInvocationID.x;");
                    remove_str(snippet, "uint c = gl_GlobalInvocationID.x;");
                    remove_str(snippet, "uint ow = gl_GlobalInvocationID.y;");
                    remove_str(snippet, "uint iw = gl_GlobalInvocationID.y;");
                    remove_str(snippet, "uint n = gl_GlobalInvocationID.y;");
                    remove_str(snippet, "uint n_oh = gl_GlobalInvocationID.z;");
                    remove_str(snippet, "uint n_ih = gl_GlobalInvocationID.z;");
                }
            }

            gen.addLogicSnippet(std::format("        {}", snippet));

            if (op_idx == 0 && !meta.index_expr.empty())
            {
                std::string resolved_expr = meta.index_expr;
                for (int pc_i = 31; pc_i >= 0; --pc_i)
                {
                    std::string token = std::format("{{pc_{}}}", pc_i);
                    std::string replacement = std::format("pc.data[{}]", pc_word_offset + pc_i);
                    std::size_t pos = 0;
                    while ((pos = resolved_expr.find(token, pos)) != std::string::npos)
                    {
                        resolved_expr.replace(pos, token.length(), replacement);
                        pos += replacement.length();
                    }
                }
                gen.addLogicSnippet(std::format("        global_id = {};", resolved_expr));
            }

            for (const auto &stmt : post_statements)
            {
                gen.addLogicSnippet(stmt);
            }
        }

        gen.addLogicSnippet("    }");

        std::string glsl_code = gen.build();
#if ENABLE_EXECUTOR_DEBUG_LOGS
        {
            std::string node_chain_name = "[";
            for (std::size_t op_i = 0; op_i < node.fused_operations.size(); ++op_i)
            {
                node_chain_name += std::string(magic_enum::enum_name(node.fused_operations[op_i].pipeline_id));
                if (op_i + 1 < node.fused_operations.size())
                {
                    node_chain_name += " -> ";
                }
            }
            node_chain_name += "]";
            if (std::find(has_print_terminal.begin(), has_print_terminal.end(), node_chain_name) == has_print_terminal.end())
            {
                has_print_terminal.push_back(node_chain_name);
                EXECUTOR_LOG_DEBUG(std::format("Graph_Executor::generateFusedGlsl: Generated GLSL for node {}:\n\n===============\n{}\n\n===============",
                                               node_chain_name, glsl_code));
            }
        }
#endif

        return glsl_code;
    }

    void executeFallbackNode(VkCommandBuffer cmd_buffer, const Compute_Node &node, std::uint32_t frame_index)
    {
        VkDevice device = context.getDevice();
        VkDescriptorSetLayout layout = network.getDescriptorSetLayout();

        for (std::size_t op_idx = 0; op_idx < node.fused_operations.size(); ++op_idx)
        {
            const auto &op = node.fused_operations[op_idx];

            std::vector<std::shared_ptr<GVector>> op_buffers;
            for (std::uint32_t idx : op.input_buffer_indices)
            {
                if (idx < node.buffers.size())
                {
                    op_buffers.push_back(node.buffers[idx]);
                }
            }
            for (std::uint32_t idx : op.output_buffer_indices)
            {
                if (idx < node.buffers.size())
                {
                    op_buffers.push_back(node.buffers[idx]);
                }
            }

            VkDescriptorSetAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            alloc_info.descriptorPool = descriptor_pools[frame_index];
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &layout;

            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) != VK_SUCCESS)
            {
                Logger::logMessage(std::format("Graph_Executor::executeFallbackNode: Failed to allocate descriptor set for fallback op {}", op_idx), LOG_ERROR, true);
                throw std::runtime_error("Failed to allocate descriptor set");
            }
            descriptor_cache[frame_index].push_back(descriptor_set);

            updateDescriptorSet(descriptor_set, op_buffers);

            VkPipeline static_pipeline = network.getPipeline(op.pipeline_id);

            vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, static_pipeline);
            vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &descriptor_set, 0, nullptr);

            if (op.pc_size > 0 && op.pc_offset + op.pc_size <= node.push_constants_data.size())
            {
                vkCmdPushConstants(
                    cmd_buffer,
                    network.getPipelineLayout(),
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    op.pc_size,
                    node.push_constants_data.data() + op.pc_offset);
            }

            vkCmdDispatch(cmd_buffer, op.group_x, op.group_y, op.group_z);

            if (op_idx < node.fused_operations.size() - 1)
            {
                insertMemoryBarrier(cmd_buffer);
            }
        }
    }

    void initResources()
    {
        EXECUTOR_LOG_DEBUG("Graph_Executor::initResources: Allocating command buffers and descriptor pools");

        VkDevice device = context.getDevice();

        VkCommandBufferAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc_info.commandPool = context.getCommandPool();
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

        if (vkAllocateCommandBuffers(device, &alloc_info, cmd_buffers) != VK_SUCCESS)
        {
            Logger::logMessage("Graph_Executor::initResources: Failed to allocate command buffers", LOG_ERROR, true);
            throw std::runtime_error("Failed to allocate command buffers");
        }

        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4000}};

        VkDescriptorPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.flags = 0;
        pool_info.maxSets = 1000;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = pool_sizes;

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pools[i]) != VK_SUCCESS)
            {
                Logger::logMessage("Graph_Executor::initResources: Failed to create descriptor pool for frame " + std::to_string(i), LOG_ERROR, true);
                throw std::runtime_error("Failed to create descriptor pool");
            }
        }
    }

    void insertMemoryBarrier(VkCommandBuffer cmd_buffer) const
    {
        VkMemoryBarrier memory_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memory_barrier,
            0, nullptr,
            0, nullptr);
    }

    void updateDescriptorSet(VkDescriptorSet descriptor_set, const std::vector<std::shared_ptr<GVector>> &buffers) const
    {
        if (buffers.empty())
        {
            Logger::logMessage("Graph_Executor::updateDescriptorSet: Node buffers vector is empty", LOG_WARNING);
            return;
        }

        VkDevice device = context.getDevice();
        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());

        for (std::size_t i = 0; i < buffers.size(); ++i)
        {
            if (!buffers[i])
            {
                Logger::logMessage("Graph_Executor::updateDescriptorSet: Null buffer encountered in node buffers at index " + std::to_string(i), LOG_WARNING);
                continue;
            }

            buffer_infos[i].buffer = buffers[i]->getBuffer();
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = VK_WHOLE_SIZE;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }

        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void updateDescriptorSet(VkDescriptorSet descriptor_set,
                             const std::vector<std::uint32_t> &binding_indices,
                             const std::vector<std::shared_ptr<GVector>> &buffers) const
    {
        if (buffers.empty() || binding_indices.size() != buffers.size())
        {
            Logger::logMessage("Graph_Executor::updateDescriptorSet: Mismatched or empty binding/buffer list", LOG_WARNING);
            return;
        }

        VkDevice device = context.getDevice();
        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());

        for (std::size_t i = 0; i < buffers.size(); ++i)
        {
            if (!buffers[i])
                continue;

            buffer_infos[i].buffer = buffers[i]->getBuffer();
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = VK_WHOLE_SIZE;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = binding_indices[i];
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }

        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

public:
    Graph_Executor(const Vulkan_Context &ctx, const Vulkan_Network &net, Pipeline_Cache_Manager &cache_mgr, const Shader_Dictionary &dict)
        : context(ctx), network(net), cache_manager(cache_mgr), shader_dict(dict), has_print_terminal(std::vector<std::string>(0))
    {
        initResources();
    }

    ~Graph_Executor()
    {
        EXECUTOR_LOG_DEBUG("Graph_Executor::~Graph_Executor: Cleaning up Vulkan descriptor pools and command buffers");
        VkDevice device = context.getDevice();
        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (descriptor_pools[i] != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, descriptor_pools[i], nullptr);
            }
        }
        vkFreeCommandBuffers(device, context.getCommandPool(), MAX_FRAMES_IN_FLIGHT, cmd_buffers);
    }

    void resetFrameState(std::uint32_t frame_index)
    {
        if (frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage("Graph_Executor::resetFrameState: frame_index out of bounds (" + std::to_string(frame_index) + ")", LOG_WARNING);
            return;
        }

        EXECUTOR_LOG_DEBUG("Graph_Executor::resetFrameState: Resetting descriptor pool for frame " + std::to_string(frame_index));
        vkResetDescriptorPool(context.getDevice(), descriptor_pools[frame_index], 0);
        descriptor_cache[frame_index].clear();
    }

    void compileAndExecute(const Compute_Graph &graph, const std::vector<Buffer_Transfer_Task> &transfers, std::uint32_t frame_index, VkFence external_fence = VK_NULL_HANDLE)
    {
        if (frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage("Graph_Executor::compileAndExecute: frame_index out of bounds (" + std::to_string(frame_index) + ")", LOG_WARNING);
            return;
        }

        const std::vector<Compute_Node> &nodes = graph.getNodes();
        if (nodes.empty() && transfers.empty())
        {
            Logger::logMessage("Graph_Executor::compileAndExecute: Both compute nodes and transfer tasks are empty for frame " + std::to_string(frame_index), LOG_WARNING);
        }

        EXECUTOR_LOG_DEBUG("Graph_Executor::compileAndExecute: Frame " + std::to_string(frame_index) + ", nodes=" + std::to_string(nodes.size()) + ", transfers=" + std::to_string(transfers.size()));

        VkDevice device = context.getDevice();

        context.resetFrameFence(frame_index);

        VkCommandBuffer cmd_buffer = cmd_buffers[frame_index];
        if (vkResetCommandBuffer(cmd_buffer, 0) != VK_SUCCESS)
        {
            Logger::logMessage("Graph_Executor::compileAndExecute: Failed to reset command buffer for frame " + std::to_string(frame_index), LOG_ERROR, true);
            throw std::runtime_error("Failed to reset command buffer");
        }

        VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd_buffer, &begin_info) != VK_SUCCESS)
        {
            Logger::logMessage("Graph_Executor::compileAndExecute: Failed to begin command buffer for frame " + std::to_string(frame_index), LOG_ERROR, true);
            throw std::runtime_error("Failed to begin command buffer");
        }

        if (!transfers.empty())
        {
            for (const auto &task : transfers)
            {
                if (task.size == 0 || task.src_buffer == VK_NULL_HANDLE || task.dst_buffer == VK_NULL_HANDLE)
                {
                    continue;
                }

                VkBufferCopy copy_region{};
                copy_region.srcOffset = task.src_offset;
                copy_region.dstOffset = task.dst_offset;
                copy_region.size = task.size;

                vkCmdCopyBuffer(cmd_buffer, task.src_buffer, task.dst_buffer, 1, &copy_region);
            }

            VkMemoryBarrier transfer_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            transfer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            transfer_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                cmd_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &transfer_barrier,
                0, nullptr,
                0, nullptr);
        }

        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            const Compute_Node &node = nodes[i];

            for (const auto &vec : node.buffers)
            {
                if (vec)
                {
                    vec->markAsUsedInFrame(frame_index);
                }
                else
                {
                    Logger::logMessage("Graph_Executor::compileAndExecute: Null GVector buffer encountered in compute node " + std::to_string(i), LOG_WARNING);
                }
            }

            bool fused_success = false;
            if (node.is_fused && node.fused_operations.size() > 1)
            {
                try
                {
                    std::string glsl_code = generateFusedGlsl(node);
                    VkPipeline target_pipeline = cache_manager.getOrCreatePipeline(glsl_code);

                    if (target_pipeline != VK_NULL_HANDLE)
                    {
                        std::vector<std::uint32_t> external_indices = getExternalBufferIndices(node);
                        std::vector<std::shared_ptr<GVector>> fused_external_buffers;
                        for (std::uint32_t buf_idx : external_indices)
                        {
                            if (buf_idx < node.buffers.size())
                            {
                                fused_external_buffers.push_back(node.buffers[buf_idx]);
                            }
                        }

                        VkDescriptorSetLayout layout = network.getDescriptorSetLayout();
                        VkDescriptorSetAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                        alloc_info.descriptorPool = descriptor_pools[frame_index];
                        alloc_info.descriptorSetCount = 1;
                        alloc_info.pSetLayouts = &layout;

                        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
                        if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) == VK_SUCCESS)
                        {
                            descriptor_cache[frame_index].push_back(descriptor_set);
                            updateDescriptorSet(descriptor_set, external_indices, fused_external_buffers);

                            vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, target_pipeline);
                            vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &descriptor_set, 0, nullptr);

                            if (!node.push_constants_data.empty())
                            {
                                vkCmdPushConstants(cmd_buffer, network.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<std::uint32_t>(node.push_constants_data.size()), node.push_constants_data.data());
                            }

                            vkCmdDispatch(cmd_buffer, node.group_x, node.group_y, node.group_z);
                            fused_success = true;
#if ENABLE_FUSION_SHADOW_VALIDATION
                            static std::uint32_t shadow_dispatch_counter = 0;
                            if (++shadow_dispatch_counter % SHADOW_SAMPLING_RATE == 0)
                            {
                                EXECUTOR_LOG_DEBUG("Graph_Executor::compileAndExecute: Triggering Shadow Validation for fused node");
                                performShadowValidation(node, target_pipeline, frame_index);
                            }
#endif
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Fused shader compilation failed for node [{}] ({}), initiating fallback execution", i, e.what()), LOG_WARNING);
                    fused_success = false;
                }
            }

            if (!fused_success)
            {
                if (node.is_fused && node.fused_operations.size() > 1)
                {
                    executeFallbackNode(cmd_buffer, node, frame_index);
                }
                else
                {
                    VkDescriptorSetLayout layout = network.getDescriptorSetLayout();
                    VkDescriptorSetAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                    alloc_info.descriptorPool = descriptor_pools[frame_index];
                    alloc_info.descriptorSetCount = 1;
                    alloc_info.pSetLayouts = &layout;

                    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
                    if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) != VK_SUCCESS)
                    {
                        Logger::logMessage("Graph_Executor::compileAndExecute: Failed to allocate descriptor set for node " + std::to_string(i), LOG_ERROR, true);
                        throw std::runtime_error("Failed to allocate descriptor set");
                    }
                    descriptor_cache[frame_index].push_back(descriptor_set);

                    updateDescriptorSet(descriptor_set, node.buffers);

                    VkPipeline target_pipeline = network.getPipeline(node.pipeline_id);

                    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, target_pipeline);
                    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &descriptor_set, 0, nullptr);

                    if (!node.push_constants_data.empty())
                    {
                        vkCmdPushConstants(cmd_buffer, network.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<std::uint32_t>(node.push_constants_data.size()), node.push_constants_data.data());
                    }

                    vkCmdDispatch(cmd_buffer, node.group_x, node.group_y, node.group_z);
                }
            }

            if (i < nodes.size() - 1)
            {
                insertMemoryBarrier(cmd_buffer);
            }
        }

        if (vkEndCommandBuffer(cmd_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("Graph_Executor::compileAndExecute: Failed to end command buffer for frame " + std::to_string(frame_index), LOG_ERROR, true);
            throw std::runtime_error("Failed to end command buffer");
        }

        VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer;

        VkFence primary_fence = (external_fence != VK_NULL_HANDLE) ? external_fence : context.getFrameFence(frame_index);

        if (vkQueueSubmit(context.getComputeQueue(), 1, &submit_info, primary_fence) != VK_SUCCESS)
        {
            Logger::logMessage("Graph_Executor::compileAndExecute: Failed to submit command buffer to compute queue for frame " + std::to_string(frame_index), LOG_ERROR, true);
            throw std::runtime_error("Failed to submit command buffer");
        }

        if (external_fence != VK_NULL_HANDLE && external_fence != context.getFrameFence(frame_index))
        {
            vkQueueSubmit(context.getComputeQueue(), 0, nullptr, context.getFrameFence(frame_index));
        }
    }

    void warmupPipelineCache(const Compute_Graph &graph)
    {
        const std::vector<Compute_Node> &nodes = graph.getNodes();
        if (nodes.empty())
        {
            return;
        }

        EXECUTOR_LOG_DEBUG(std::format("Graph_Executor::warmupPipelineCache: Pre-compiling pipelines for {} nodes", nodes.size()));

        for (const auto &node : nodes)
        {
            if (node.is_fused && node.fused_operations.size() > 1)
            {
                try
                {
                    std::string glsl_code = generateFusedGlsl(node);
                    cache_manager.getOrCreatePipeline(glsl_code);
                }
                catch (const std::exception &e)
                {
                    Logger::logMessage(std::format("Graph_Executor::warmupPipelineCache: Warmup pre-compilation failed: {}, fallback will be used at runtime", e.what()), LOG_WARNING);
                }
            }
            else
            {
                network.getPipeline(node.pipeline_id);
            }
        }
    }

#if ENABLE_FUSION_SHADOW_VALIDATION
    void performShadowValidation(const Compute_Node &node, VkPipeline fused_pipeline, std::uint32_t frame_index)
    {
        std::string node_chain_name;
        if (!node.fused_operations.empty())
        {
            for (std::size_t i = 0; i < node.fused_operations.size(); ++i)
            {
                if (i > 0)
                {
                    node_chain_name += " -> ";
                }
                node_chain_name += magic_enum::enum_name(node.fused_operations[i].pipeline_id);
            }
        }
        else
        {
            node_chain_name = std::string(magic_enum::enum_name(node.pipeline_id));
        }

        EXECUTOR_LOG_DEBUG(std::format("Graph_Executor::performShadowValidation: Starting validation for node [{}] (ops_count={}, grid=[{}, {}, {}])",
                                       node_chain_name, node.fused_operations.size(), node.group_x, node.group_y, node.group_z));

        VkDevice device = context.getDevice();
        VkCommandPool command_pool = context.getCommandPool();
        VkQueue compute_queue = context.getComputeQueue();

        VkCommandBufferAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer shadow_cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device, &alloc_info, &shadow_cmd);

        VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(shadow_cmd, &begin_info);

        std::vector<std::shared_ptr<GVector>> fused_buffers;
        std::vector<std::shared_ptr<GVector>> fallback_buffers;

        for (const auto &buf : node.buffers)
        {
            if (buf)
            {
                auto fused_clone = std::make_shared<GVector>(context, buf->getSize());
                auto fallback_clone = std::make_shared<GVector>(context, buf->getSize());

                VkBufferCopy copy_region{};
                copy_region.size = buf->getSizeBytes();

                vkCmdCopyBuffer(shadow_cmd, buf->getBuffer(), fused_clone->getBuffer(), 1, &copy_region);
                vkCmdCopyBuffer(shadow_cmd, buf->getBuffer(), fallback_clone->getBuffer(), 1, &copy_region);

                fused_buffers.push_back(fused_clone);
                fallback_buffers.push_back(fallback_clone);
            }
            else
            {
                fused_buffers.push_back(nullptr);
                fallback_buffers.push_back(nullptr);
            }
        }

        VkMemoryBarrier copy_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        copy_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copy_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(shadow_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &copy_barrier, 0, nullptr, 0, nullptr);

        Compute_Node fused_node = node;
        fused_node.buffers = fused_buffers;

        Compute_Node fallback_node = node;
        fallback_node.buffers = fallback_buffers;

        std::vector<std::uint32_t> external_indices = getExternalBufferIndices(fused_node);
        std::vector<std::shared_ptr<GVector>> fused_external_buffers;
        for (std::uint32_t buf_idx : external_indices)
        {
            if (buf_idx < fused_node.buffers.size())
            {
                fused_external_buffers.push_back(fused_node.buffers[buf_idx]);
            }
        }

        VkDescriptorSetLayout layout = network.getDescriptorSetLayout();
        VkDescriptorSetAllocateInfo set_alloc_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_alloc_info.descriptorPool = descriptor_pools[frame_index];
        set_alloc_info.descriptorSetCount = 1;
        set_alloc_info.pSetLayouts = &layout;

        VkDescriptorSet fused_set = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(device, &set_alloc_info, &fused_set);
        descriptor_cache[frame_index].push_back(fused_set);

        updateDescriptorSet(fused_set, external_indices, fused_external_buffers);

        vkCmdBindPipeline(shadow_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fused_pipeline);
        vkCmdBindDescriptorSets(shadow_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &fused_set, 0, nullptr);

        if (!fused_node.push_constants_data.empty())
        {
            vkCmdPushConstants(shadow_cmd, network.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<std::uint32_t>(fused_node.push_constants_data.size()), fused_node.push_constants_data.data());
        }

        vkCmdDispatch(shadow_cmd, fused_node.group_x, fused_node.group_y, fused_node.group_z);
        insertMemoryBarrier(shadow_cmd);

        executeFallbackNode(shadow_cmd, fallback_node, frame_index);

        std::unordered_set<std::uint32_t> output_indices_to_verify(node.external_output_indices.begin(), node.external_output_indices.end());
        if (!node.fused_operations.empty())
        {
            for (std::uint32_t out_idx : node.fused_operations.back().output_buffer_indices)
            {
                output_indices_to_verify.insert(out_idx);
            }
        }

        std::vector<VkBuffer> fused_staging_bufs(fused_buffers.size(), VK_NULL_HANDLE);
        std::vector<VkBuffer> fallback_staging_bufs(fallback_buffers.size(), VK_NULL_HANDLE);
        std::vector<Memory_Allocation> fused_staging_allocs(fused_buffers.size());
        std::vector<Memory_Allocation> fallback_staging_allocs(fallback_buffers.size());

        VkMemoryBarrier readback_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        readback_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        readback_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(shadow_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readback_barrier, 0, nullptr, 0, nullptr);

        for (std::uint32_t b : output_indices_to_verify)
        {
            if (b < fused_buffers.size() && fused_buffers[b] && fallback_buffers[b])
            {
                std::size_t buf_size = fused_buffers[b]->getSizeBytes();

                VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                buffer_info.size = buf_size;
                buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                vkCreateBuffer(device, &buffer_info, nullptr, &fused_staging_bufs[b]);
                vkCreateBuffer(device, &buffer_info, nullptr, &fallback_staging_bufs[b]);

                VkMemoryRequirements mem_reqs;
                vkGetBufferMemoryRequirements(device, fused_staging_bufs[b], &mem_reqs);

                fused_staging_allocs[b] = context.allocateMemory(mem_reqs, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                fallback_staging_allocs[b] = context.allocateMemory(mem_reqs, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                vkBindBufferMemory(device, fused_staging_bufs[b], fused_staging_allocs[b].memory, fused_staging_allocs[b].offset);
                vkBindBufferMemory(device, fallback_staging_bufs[b], fallback_staging_allocs[b].memory, fallback_staging_allocs[b].offset);

                VkBufferCopy copy_region{};
                copy_region.size = buf_size;
                vkCmdCopyBuffer(shadow_cmd, fused_buffers[b]->getBuffer(), fused_staging_bufs[b], 1, &copy_region);
                vkCmdCopyBuffer(shadow_cmd, fallback_buffers[b]->getBuffer(), fallback_staging_bufs[b], 1, &copy_region);
            }
        }

        vkEndCommandBuffer(shadow_cmd);

        VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &shadow_cmd;
        vkQueueSubmit(compute_queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(compute_queue);

        vkFreeCommandBuffers(device, command_pool, 1, &shadow_cmd);

        bool is_validation_passed = true;
        std::size_t tested_buffers_count = 0;

        for (std::uint32_t b : output_indices_to_verify)
        {
            if (b < fused_buffers.size() && fused_staging_bufs[b] != VK_NULL_HANDLE && fallback_staging_bufs[b] != VK_NULL_HANDLE)
            {
                tested_buffers_count++;
                std::size_t buf_size = fused_buffers[b]->getSizeBytes();
                void *fused_ptr = nullptr;
                void *fallback_ptr = nullptr;

                vkMapMemory(device, fused_staging_allocs[b].memory, fused_staging_allocs[b].offset, buf_size, 0, &fused_ptr);
                vkMapMemory(device, fallback_staging_allocs[b].memory, fallback_staging_allocs[b].offset, buf_size, 0, &fallback_ptr);

                float *fused_data = static_cast<float *>(fused_ptr);
                float *fallback_data = static_cast<float *>(fallback_ptr);
                std::size_t element_count = buf_size / sizeof(float);

                std::size_t err_count = 0;
                float max_err = 0.0f;

                for (std::size_t i = 0; i < element_count; ++i)
                {
                    float diff = std::abs(fallback_data[i] - fused_data[i]);
                    if (diff > SHADOW_EPSILON)
                    {
                        err_count++;
                        max_err = std::max(max_err, diff);
                    }
                }

                EXECUTOR_LOG_DEBUG(std::format("Graph_Executor::performShadowValidation: Node [{}] Buffer [{}] ({} elements, {} bytes) | errors={}/{} | max_diff={:.6e} | threshold={:.1e}",
                                               node_chain_name, b, element_count, buf_size, err_count, element_count, max_err, static_cast<float>(SHADOW_EPSILON)));

                if (err_count > 0)
                {
                    is_validation_passed = false;
                    std::string msg = std::format("Shadow Validation Failed: Node [{}] Buffer [{}]. Mismatches: {}/{}. Max Error: {:.6e}",
                                                  node_chain_name, b, err_count, element_count, max_err);
                    Logger::logMessage(msg, LOG_ERROR, true);
                }

                vkUnmapMemory(device, fused_staging_allocs[b].memory);
                vkUnmapMemory(device, fallback_staging_allocs[b].memory);

                vkDestroyBuffer(device, fused_staging_bufs[b], nullptr);
                vkDestroyBuffer(device, fallback_staging_bufs[b], nullptr);
                context.deferDestruction(frame_index, VK_NULL_HANDLE, fused_staging_allocs[b]);
                context.deferDestruction(frame_index, VK_NULL_HANDLE, fallback_staging_allocs[b]);
            }
        }

        if (is_validation_passed)
        {
            EXECUTOR_LOG_DEBUG(std::format("Graph_Executor::performShadowValidation: SUCCESS - Node [{}] ({} tested buffers) matched within tolerance ({:.1e})",
                                           node_chain_name, tested_buffers_count, static_cast<float>(SHADOW_EPSILON)));
        }
        else
        {
            Logger::logMessage(std::format("Graph_Executor::performShadowValidation: FAILED - Node [{}] Numerical divergence detected", node_chain_name), LOG_ERROR, true);
        }
    }
#endif
};