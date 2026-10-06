# VulkanML — Sơ đồ Kiến trúc UML Chi tiết

## 1. Architecture Layering Diagram (Phân tầng Kiến trúc)

```mermaid
graph TD
    subgraph L6["Layer 6 — Application & High-Level Models"]
        NN["Neural_Network\n(Sequential Model, Training Loop & Checkpoints)"]
        POP["Population\n(Neuroevolution, SoA Batched GPU GEMM)"]
        TC["Training_Context\n(Epoch State, Optimizer, Loss & Scheduler Management)"]
        LLM["Causal_LM\n(Decoder-only Transformer LLM & Generation Loop)"]
        RL["RL Agents\n(DQN Agent, PPO Agent, Replay Buffer, CartPole Env)"]
    end

    subgraph L5["Layer 5 — Tokenizers & Phonetics"]
        BPE["Bpe_Tokenizer\n(Byte-Pair Encoding, HuggingFace JSON Compatible)"]
        SYL["Syllable_Tokenizer\n(Vietnamese Syllable Segmentation & Rhyme Matching)"]
        VNP["Vietnamese_Phonetics\n(Tone Analysis: Bằng/Trắc, Rhyme Family)"]
    end

    subgraph L4["Layer 4 — Concrete Neural Layers & Operators"]
        LIN["Linear_Layer\n(Dense GEMM + Adam Fused Backward)"]
        CONV["Conv2d_Layer\n(Im2Col / Direct GEMM)"]
        BN["BatchNorm 1D & 2D\n(Running Mean/Variance)"]
        RMSN["RMSNorm_Layer\n(Root Mean Square Normalization)"]
        SWIGLU["SwiGLU_Layer\n(Swish-Gated Activation Layer)"]
        EMB["Embedding_Layer\n(Token Lookup & Grad Accumulation)"]
        TRB["Transformer_Block\n(Pre-RMSNorm + RoPE + FlashAttention + SwiGLU FFN)"]
        ACT["Activation Layers\n(ReLU, GELU, Softmax)"]
        POOL["Pooling Layers\n(MaxPool2D, GlobalAvgPool2D)"]
        RES["Residual Networks\n(Res_Net_Block_2d_Layer, Res_Net_20_Layer)"]
        PPO_L["PPO_Actor_Critic_Layer\n(Actor Policy + Critic Value Heads)"]
    end

    subgraph L4i["Layer 4 — Loss Functions, Optimizers & Schedulers"]
        COST["ICost_Function\n(MSE, MAE, BCE, CCE, Huber, Fused_Cross_Entropy)"]
        OPT["IOptimizer\n(Adam_Optimizer with AMP & Weight Decay, Sgd_Optimizer)"]
        LRS["ILearning_Rate\n(Cosine_Annealing, Step, Exponential, Polynomial, Plateau)"]
        IL["ILayer\n(Abstract Base Class)"]
    end

    subgraph L3["Layer 3 — Tensor & Math Abstraction System"]
        TENS["Tensor\n(Public PIMPL API, Strides, Slicing & Serialization)"]
        SHP["Shape & Stride\n(Static Array Dims up to Rank 6)"]
        CPU_IMPL["Cpu_Tensor_Impl\n(AVX/CPU Math Implementations)"]
        GPU_IMPL["Gpu_Tensor_Impl\n(Vulkan Compute Kernels, FlashAttention, RoPE)"]
        TIMPL["Tensor_Impl\n(Abstract Interface)"]
    end

    subgraph L2["Layer 2 — Graph Optimization & Execution Engine"]
        EE["Execution_Engine\n(Singleton, Graph Cache, Double-Buffered Frames)"]
        GO["Graph_Optimizer\n(Static JIT Fusion Validation & Hazard Analysis)"]
        GX["Graph_Executor\n(Vulkan Command Dispatch, Static Buffers, Fallback)"]
        CG["Compute_Graph\n(DAG Container of Compute_Node)"]
        LS["Loss_Scaler\n(Dynamic Mixed Precision FP16 AMP)"]
        ADP["Async_Data_Pipeline\n(Double-Buffered Background Disk I/O)"]
    end

    subgraph L1["Layer 1 — Vulkan Core Compute Backend"]
        VK["Vulkan_Context\n(Physical/Logical Device, Compute Queue, Staging Buffers)"]
        NET["Vulkan_Network\n(Pipeline Layout, Descriptor Layout & Static Pipelines)"]
        ALLOC["Vulkan_Sub_Allocator\n(Device Memory Chunk Pool & Dynamic Blocks)"]
        GPUV["gpu_vector\n(RAII GPU Buffer with Sub-Allocation)"]
        SHAD["Shader_Dictionary\n(Snippet JSON Metadata & Fusion Capabilities)"]
        SHGEN["Shader_Generator\n(Dynamic GLSL Stitching for Operator Fusion)"]
        SHCOMP["Shader_Compiler\n(glslc / libshaderc SPIR-V Compilation)"]
        PCM["Pipeline_Cache_Manager\n(On-disk & In-memory VkPipeline Cache)"]
    end

    subgraph L0["Layer 0 — Foundation Utilities & Profiling"]
        LOG["Logger\n(Feature-Masked Logging & Timestamps)"]
        UPREFS["User_Preferences\n(JSON Preferences & Device Config)"]
        PROF["Training_Profiler\n(Stage Timing Accumulators)"]
    end

    L6 --> L5
    L6 --> L4
    L6 --> L4i
    L4 --> L3
    L4i --> L3
    L3 --> L2
    L2 --> L1
    L1 --> L0
```

---

## 2. Class Diagram — Tensor System (Hệ thống Tensor)

```mermaid
classDiagram
    class Shape {
        -array~size_t~ dimensions
        -uint8_t rank_size
        +Shape()
        +Shape(dim_0, dim_1)
        +getRank() size_t
        +getTotalElements() size_t
        +computeContiguousStrides() Shape
        +toString() string
        +getDimensions() span~size_t~
        +operatorIndex(index) size_t
    }

    class Tensor_Impl {
        <<abstract>>
        #Shape shape
        #Shape strides
        #size_t byte_offset
        #size_t total_elements
        #Data_Type data_type
        +getShape() Shape
        +getStrides() Shape
        +getRows() size_t
        +getColumns() size_t
        +getDataType() Data_Type
        +isContiguous() bool
        +validateSameDimensions(other)
        +validateMatmulDimensions(other)
        +validateSquare()
        +matmul(other, output)*
        +add(other, output)*
        +relu(output)*
        +conv2d(weights, biases, output)*
        +flashAttentionForward(k, v, output)*
        +fusedCrossEntropyLoss(targets, d_logits)* float
        +adamUpdate(gradient, m, v, lr, beta1, beta2, eps, step)*
    }

    class Cpu_Tensor_Impl {
        -vector~float~ data
        +getData() vector~float~
        +uploadData(host_data)
        +matmul(other, output)
        +add(other, output)
        +relu(output)
        +conv2d(weights, biases, output)
        +flashAttentionForward(k, v, output)
        +fusedCrossEntropyLoss(targets, d_logits) float
        +adamUpdate(gradient, m, v, lr, beta1, beta2, eps, step)
    }

    class Gpu_Tensor_Impl {
        -shared_ptr~gpu_vector~ gpu_vec
        -shared_ptr~gpu_vector~ fp16_gpu_vec
        +getVector() shared_ptr~gpu_vector~
        +uploadData(host_data)
        +matmul(other, output)
        +add(other, output)
        +relu(output)
        +conv2d(weights, biases, output)
        +flashAttentionForward(k, v, output)
        +fusedCrossEntropyLoss(targets, d_logits) float
        +adamUpdate(gradient, m, v, lr, beta1, beta2, eps, step)
    }

    class Tensor {
        -shared_ptr~Tensor_Impl~ implementation
        -Execution_Target execution_target
        +Tensor(target)
        +Tensor(rows, cols, target)
        +Tensor(rows, cols, host_data, target)
        +Tensor(shape, target)
        +matmul(other, output)
        +matmulAdd(other, biases) Tensor
        +operatorPlus(other) Tensor
        +relu() Tensor
        +to(target_type) Tensor
        +toFp16() Tensor
        +toFp32() Tensor
        +saveTensor(output_stream)
        +loadTensor(input_stream, target)$ Tensor
        +getRows() size_t
        +getColumns() size_t
        +getShape() Shape
        +getData() vector~float~
        +getStorage() Storage_Handle
        +getExecutionTarget() Execution_Target
        +setExecutionTarget(new_target)
    }

    class gpu_vector {
        -Vulkan_Context context
        -VkBuffer buffer
        -Memory_Allocation allocation
        -size_t buffer_size_in_bytes
        -size_t element_count
        -Data_Type data_type
        +getBuffer() VkBuffer
        +getSize() size_t
        +uploadData(host_data)
        +download() vector~float~
        +allocateMemory(count, type)
        +freeMemory()
        +isEmpty() bool
    }

    Tensor_Impl <|-- Cpu_Tensor_Impl
    Tensor_Impl <|-- Gpu_Tensor_Impl
    Tensor o-- Tensor_Impl : implementation
    Gpu_Tensor_Impl o-- gpu_vector : gpu_vec
    Shape <-- Tensor_Impl : shape and strides
```

---

## 3. Class Diagram — Layer System (Hệ thống Lớp Nơ-ron)

```mermaid
classDiagram
    class ILayer {
        <<abstract>>
        #bool is_accumulated
        #bool is_mixed_precision_enabled
        +forward(input_tensor) Tensor
        +backward(output_gradient)* Tensor
        +clone()* unique_ptr~ILayer~
        +saveConfiguration(output_stream)*
        +saveInference(output_stream)*
        +loadInference(input_stream)*
        +saveCheckpoint(output_stream)*
        +loadCheckpoint(input_stream)*
        +getLayerType()* Layer_Type
        +getExecutionTarget()* Execution_Target
        +setExecutionTarget(target)*
        +hasParameters() bool
        +getParametersAndGradients() vector~pair~
        +setMixedPrecision(enable)
        +setAccumulated(is_accumulated)
        +logBufferAddress(tensor, name)
    }

    class Linear_Layer {
        -Tensor weights
        -Tensor biases
        -Tensor weights_gradient
        -Tensor bias_gradient
        -Tensor input_cache
        +forward(input_tensor) Tensor
        +backward(output_gradient) Tensor
        +saveInference(output_stream)
        +loadInference(input_stream)
        +getLayerType() Layer_Type
    }

    class Conv2d_Layer {
        -Tensor weights
        -Tensor biases
        -uint32_t kernel_size
        -uint32_t stride
        -uint32_t padding
        +forward(input_tensor) Tensor
        +backward(output_gradient) Tensor
    }

    class Transformer_Block {
        -size_t hidden_dim
        -size_t num_heads
        -size_t head_dim
        -RMSNorm_Layer input_layernorm
        -Linear_Layer q_proj
        -Linear_Layer k_proj
        -Linear_Layer v_proj
        -Linear_Layer o_proj
        -RMSNorm_Layer post_attention_layernorm
        -Linear_Layer gate_proj
        -Linear_Layer up_proj
        -Linear_Layer down_proj
        -SwiGLU_Layer swiglu
        +forward(input_tensor) Tensor
        +forward(input_tensor, kv_cache) Tensor
        +backward(output_gradient) Tensor
    }

    class RMSNorm_Layer {
        -Tensor gamma
        -Tensor inv_rms_cache
        -float epsilon
        +forward(input_tensor) Tensor
        +backward(output_gradient) Tensor
    }

    class SwiGLU_Layer {
        -Tensor tensor_a
        -Tensor tensor_b
        -Tensor grad_a
        -Tensor grad_b
        +forward(input_tensor) Tensor
        +forward(a, b) Tensor
        +backward(output_gradient) Tensor
    }

    class Embedding_Layer {
        -Tensor weights
        -Tensor indices_cache
        +forward(indices_tensor) Tensor
        +backward(output_gradient) Tensor
    }

    class Batch_Norm_Layer {
        -Tensor gamma
        -Tensor beta
        -Tensor running_mean
        -Tensor running_variance
        +forward(input_tensor) Tensor
        +backward(output_gradient) Tensor
    }

    class Res_Net_Block_2d_Layer {
        -vector~ILayer*~ main_branch
        -vector~ILayer*~ shortcut_branch
        -unique_ptr~ILayer~ post_activation
        +addMainLayer(args)
        +addShortcutLayer(args)
        +forward(input_tensor) Tensor
        +backward(output_gradient) Tensor
    }

    ILayer <|-- Linear_Layer
    ILayer <|-- Conv2d_Layer
    ILayer <|-- Transformer_Block
    ILayer <|-- RMSNorm_Layer
    ILayer <|-- SwiGLU_Layer
    ILayer <|-- Embedding_Layer
    ILayer <|-- Batch_Norm_Layer
    ILayer <|-- Res_Net_Block_2d_Layer
    ILayer <|-- Relu_Layer
    ILayer <|-- Gelu_Layer
    ILayer <|-- Softmax_Layer
    ILayer <|-- Max_Pool_2d_Layer
    ILayer <|-- Batch_Norm_2d_Layer
    ILayer <|-- PPO_Actor_Critic_Layer
    Transformer_Block o-- RMSNorm_Layer : input and post norm
    Transformer_Block o-- SwiGLU_Layer : swiglu activation
    Transformer_Block o-- Linear_Layer : projection layers
    Res_Net_Block_2d_Layer o-- ILayer : main and shortcut branches
```

---

## 4. Class Diagram — Execution Engine & Graph Optimization (Đồ thị & Tối ưu hóa)

```mermaid
classDiagram
    class Execution_Engine {
        <<singleton>>
        -unique_ptr~Vulkan_Context~ context
        -unique_ptr~Vulkan_Network~ network
        -unique_ptr~Pipeline_Cache_Manager~ pipeline_cache_manager
        -unique_ptr~Shader_Dictionary~ shader_dictionary
        -unique_ptr~Graph_Executor~ graph_executor
        -Compute_Graph current_graph
        -unordered_map~size_t, Cached_Graph_Template~ cached_graph_templates
        -bool is_graph_cache_enabled
        -bool is_static_graph_enabled
        +getInstance()$ Execution_Engine&
        +executeGraph()
        +warmCache(raw_graph)
        +getCurrentGraph() Compute_Graph&
        +isCooperativeMatrixSupported() bool
        +setCooperativeMatrixEnabled(enable)
        +precompileTemplatePipelines(template)
    }

    class Compute_Graph {
        -vector~Compute_Node~ nodes
        +addNode(node)
        +getNodeCount() size_t
        +getNodes() vector~Compute_Node~&
        +clear()
    }

    class Compute_Node {
        +Compute_Pipeline pipeline_id
        +vector~gpu_vector*~ buffers
        +vector~uint8_t~ push_constants_data
        +uint32_t workgroup_count_x
        +uint32_t workgroup_count_y
        +uint32_t workgroup_count_z
        +bool is_fused
        +vector~Fused_Operation~ fused_operations
        +string fused_glsl_code
        +VkPipeline cached_pipeline
    }

    class Fused_Operation {
        +Compute_Pipeline pipeline_id
        +vector~uint32_t~ input_buffer_indices
        +vector~uint32_t~ output_buffer_indices
        +uint32_t push_constants_offset
        +uint32_t push_constants_size
    }

    class Graph_Optimizer {
        <<utility>>
        -size_t MAX_PUSH_CONSTANTS_BYTES$
        -size_t MAX_STORAGE_BUFFER_BINDINGS$
        -size_t MAX_FUSED_OPERATIONS$
        +optimize(graph)$
        +buildCachedTemplate(graph)$ Cached_Graph_Template
        +applyCachedTemplateInPlace(raw, template, target)$
        -isFusible(prod_class, cons_class, prod_pipe, cons_pipe)$ bool
        -hasCompatibleDimensions(prod, cons, prod_cls, cons_cls)$ bool
        -hasAliasingHazard(fused_node, next_node, dict)$ bool
        -optimizeInternal(nodes, track_mappings)$ Cached_Graph_Template
    }

    class Graph_Executor {
        -Vulkan_Context& context
        -Vulkan_Network& network
        -Pipeline_Cache_Manager& pipeline_cache_manager
        -Shader_Dictionary& shader_dictionary
        +executeNode(node, cmd_buffer)
        +generateFusedGlsl(fused_node) string
        +getExternalBufferIndices(node, indices)
        +executeFallbackNode(node, cmd_buffer)
        +isBuffersMatching(entry, buffers) bool
    }

    class Cached_Graph_Template {
        +vector~Compute_Node~ fused_nodes
        +vector2D~Buffer_Binding_Mapping~ buffer_mappings
        +vector2D~Push_Constant_Mapping~ push_constants_mappings
        +vector2D~uint32_t~ raw_node_indices
        +array~Compute_Graph, 2~ instantiated_graphs
        +bool is_valid
        +isValid() bool
    }

    class Pipeline_Cache_Manager {
        -Vulkan_Context& context
        -VkPipelineLayout pipeline_layout
        -unordered_map~string, VkPipeline~ pipeline_cache
        +getOrCreatePipeline(glsl_code) VkPipeline
        +initializePipelineCache(path)
        +savePipelineCache()
    }

    Execution_Engine o-- Compute_Graph : current_graph
    Execution_Engine o-- Graph_Executor : graph_executor
    Execution_Engine o-- Pipeline_Cache_Manager : pipeline_cache_manager
    Execution_Engine o-- Cached_Graph_Template : cached_graph_templates
    Compute_Graph o-- Compute_Node : contains
    Compute_Node o-- Fused_Operation : contains
    Graph_Optimizer ..> Cached_Graph_Template : constructs
    Graph_Optimizer ..> Compute_Node : inspects and fuses
    Graph_Executor ..> Compute_Node : dispatches
```

---

## 5. Class Diagram — Vulkan Core Engine (Tầng Lõi Vulkan)

```mermaid
classDiagram
    class Vulkan_Context {
        -VkInstance instance
        -VkPhysicalDevice physical_device
        -VkDevice device
        -VkQueue compute_queue
        -VkCommandPool command_pool
        -bool is_cooperative_matrix_supported
        -bool is_float16_supported
        -unique_ptr~Vulkan_Sub_Allocator~ allocator
        -array~VkBuffer~ staging_buffers
        -array~Memory_Allocation~ staging_allocations
        +getDevice() VkDevice
        +getComputeQueue() VkQueue
        +getAllocator() Vulkan_Sub_Allocator&
        +submitCompute(command_buffer)
        +prepareFrame(frame_index)
        +isCooperativeMatrixEnabled() bool
        +isFloat16Enabled() bool
        +executePendingTransfers()
    }

    class Vulkan_Sub_Allocator {
        -vector~Memory_Chunk~ memory_chunks
        -Memory_Planner memory_planner
        +allocate(size_bytes, alignment) Memory_Allocation
        +free(allocation)
        +collectGarbage(frame_index)
        +planStaticLayout(tensor_lifetimes)
    }

    class Memory_Chunk {
        -VkDeviceMemory device_memory
        -VkBuffer buffer
        -size_t total_capacity
        -vector~Free_Block~ free_blocks
    }

    class Vulkan_Network {
        -Vulkan_Context& context
        -VkPipelineLayout pipeline_layout
        -VkDescriptorSetLayout descriptor_set_layout
        -unordered_map~Compute_Pipeline, VkPipeline~ static_pipelines
        +getPipeline(pipeline_id) VkPipeline
        +getPipelineLayout() VkPipelineLayout
        +getDescriptorSetLayout() VkDescriptorSetLayout
    }

    class Shader_Dictionary {
        <<singleton>>
        -unordered_map~Compute_Pipeline, Snippet_Metadata~ metadata_map
        +getInstance()$ Shader_Dictionary&
        +getMetadata(pipeline_id) Snippet_Metadata&
        +loadFromJson(path)
    }

    class Snippet_Metadata {
        +Compute_Pipeline pipeline_id
        +Operation_Class operation_class
        +uint32_t input_count
        +uint32_t output_count
        +uint32_t shared_memory_size
        +bool is_cooperative_matrix_support
        +bool is_writing_multiple_elements
    }

    class Shader_Generator {
        -stringstream binding_stream
        -stringstream specialization_stream
        -uint32_t current_binding
        +addBuffer(binding, name, type, access) string
        +addSpecializationConstant(id, name, type, val)
        +generateGlsl(operations) string
    }

    class Shader_Compiler {
        +compileGlslToSpirv(glsl_code)$ vector~uint32_t~
    }

    class Loss_Scaler {
        -float scale_factor
        -float max_scale
        -float growth_factor
        -float backoff_factor
        -uint32_t growth_interval
        -uint32_t steps_since_last_overflow
        +getScale() float
        +step(overflow_detected)
        +isOverflow() bool
    }

    Vulkan_Context o-- Vulkan_Sub_Allocator : allocator
    Vulkan_Sub_Allocator o-- Memory_Chunk : contains
    Vulkan_Network o-- Shader_Dictionary
    Shader_Dictionary o-- Snippet_Metadata : contains
    Shader_Generator ..> Shader_Compiler : produces GLSL
    Shader_Compiler ..> Vulkan_Network : produces SPIR-V
```

---

## 6. Class Diagram — Neural Network & Training Context (Mô hình & Quản lý Huấn luyện)

```mermaid
classDiagram
    class Neural_Network {
        -vector~ILayer*~ layers
        -Training_Context training_context
        -Loss_Scaler loss_scaler
        -Tensor last_prediction
        -Execution_Target execution_target
        -bool is_training_mode
        -bool is_mixed_precision_enabled
        -bool is_gradient_accumulation_enabled
        +addLayer(layer)
        +forward(input_tensor) Tensor
        +backward(target_or_grad) Tensor
        +trainStep(input, target) float
        +fit(dataset, epochs)
        +compileAndWarmup(batch_size, in_features, out_features)
        +saveInference(file_path)
        +loadInference(file_path)
        +saveTrainingCheckpoint(file_path)
        +loadTrainingCheckpoint(file_path)
        +setMixedPrecision(enable)
        +getTrainingContext() Training_Context&
    }

    class Training_Context {
        -size_t current_epoch
        -unique_ptr~IOptimizer~ optimizer
        -unique_ptr~ILearning_Rate~ learning_rate_scheduler
        -unique_ptr~ICost_Function~ cost_function
        +constructLayerFromConfig(stream, type, target)$ unique_ptr~ILayer~
        +setOptimizer(optimizer)
        +setCostFunction(cost_function)
        +setLearningRate(scheduler)
        +getOptimizer() IOptimizer&
        +getCostFunction() ICost_Function&
        +getLearningRateScheduler() ILearning_Rate&
        +loadHeader(stream, target) bool
    }

    class IOptimizer {
        <<abstract>>
        +step(parameter_gradient_pairs)*
        +step(parameter_gradient_pairs, grad_scale)
        +stepDynamicParams(grad_scale)
        +saveCheckpoint(output_stream)*
        +loadCheckpoint(input_stream, target)*
        +getLearningRate()* float
        +setLearningRate(learning_rate)*
        +getType()* Optimizer_Type
    }

    class Adam_Optimizer {
        -float learning_rate
        -float beta1
        -float beta2
        -float epsilon
        -float weight_decay
        -size_t timestep
        -unordered_map~Tensor*, Parameter_State~ parameter_states
        +step(parameter_gradient_pairs)
        +saveCheckpoint(output_stream)
        +loadCheckpoint(input_stream, target)
    }

    class Sgd_Optimizer {
        -float learning_rate
        -float momentum
        -float max_gradient
        +step(parameter_gradient_pairs)
    }

    class ILearning_Rate {
        <<abstract>>
        +updateRate()* float
        +step(current_value)*
        +getCurrentRate()* float
        +saveCheckpoint(output_stream)*
        +loadCheckpoint(input_stream)*
        +getType()* Decay_Mode
    }

    class ICost_Function {
        <<abstract>>
        +computeLoss(prediction, target)* float
        +computeGradient(prediction, target)* Tensor
        +computeLossAndGradient(pred, target, grad_out) float
        +isFused() bool
        +supportsIntegerTargets() bool
        +getType()* Cost_Type
    }

    class Fused_Cross_Entropy {
        -int32_t ignore_index
        +computeLoss(prediction, target) float
        +computeGradient(prediction, target) Tensor
        +computeLossAndGradient(prediction, target_indices, grad_out) float
        +isFused() bool
        +supportsIntegerTargets() bool
        +getType() Cost_Type
    }

    Neural_Network o-- ILayer : layers
    Neural_Network o-- Training_Context : training_context
    Neural_Network o-- Loss_Scaler : loss_scaler
    Training_Context o-- IOptimizer : optimizer
    Training_Context o-- ICost_Function : cost_function
    Training_Context o-- ILearning_Rate : learning_rate_scheduler
    IOptimizer <|-- Adam_Optimizer
    IOptimizer <|-- Sgd_Optimizer
    ICost_Function <|-- Fused_Cross_Entropy
    ICost_Function <|-- Mse_Cost
    ICost_Function <|-- Mae_Cost
    ICost_Function <|-- Bce_Cost
    ICost_Function <|-- Cce_Cost
    ICost_Function <|-- Huber_Cost
    ILearning_Rate <|-- Cosine_Annealing
    ILearning_Rate <|-- Step_Decay
    ILearning_Rate <|-- Multi_Step_Decay
    ILearning_Rate <|-- Exponential_Decay
    ILearning_Rate <|-- Polynomial_Decay
    ILearning_Rate <|-- Reduce_On_Plateau
    ILearning_Rate <|-- No_Decay
```

---

## 7. Class Diagram — LLM & Tokenizer Subsystem (Phân hệ Ngôn ngữ & Tokenizer)

```mermaid
classDiagram
    class Causal_LM_Config {
        +size_t vocab_size
        +size_t hidden_dim
        +size_t num_heads
        +size_t intermediate_dim
        +size_t num_layers
        +size_t max_seq_len
        +float rms_norm_eps
        +float rope_base
        +int32_t ignore_index
        +Execution_Target execution_target
        +Data_Type data_type
        +bool use_loss_scaler
        +float initial_loss_scale
        +string tokenizer_path
    }

    class Causal_LM {
        -Causal_LM_Config config
        -Embedding_Layer token_embedding
        -vector~Transformer_Block*~ blocks
        -vector~KV_Cache_Manager*~ kv_caches
        -RMSNorm_Layer final_norm
        -Linear_Layer lm_head
        -Bpe_Tokenizer tokenizer
        -unique_ptr~Loss_Scaler~ loss_scaler
        -unique_ptr~ICost_Function~ cost_function
        +forward(input_tensor, use_cache) Tensor
        +forward(token_ids, use_cache) Tensor
        +backward(d_logits, defer_execution) Tensor
        +trainStep(input_tokens, target_tokens, optimizer, max_grad_norm) float
        +forwardLossAndBackward(inputs, targets, loss_scale) float
        +generate(prompt, max_tokens, temp, top_p, eos, skip, cb, rep_pen, top_k, stops) string
        +stepOptimizer(optimizer, max_grad_norm) bool
        +resetKVCaches()
        +saveCheckpoint(file_path)
        +loadCheckpoint(file_path)
        +saveInference(file_path)
        +loadInference(file_path)
    }

    class KV_Cache_Manager {
        -size_t batch_size
        -size_t num_heads
        -size_t max_seq_len
        -size_t head_dim
        -size_t current_seq_len
        -KV_Cache_Mode mode
        -Tensor k_cache
        -Tensor v_cache
        +append(k_new, v_new)
        +appendAt(k_new, v_new, pos)
        +getK(len) Tensor
        +getV(len) Tensor
        +reset()
        +getCurrentSeqLen() size_t
        +getRawKCache() Tensor&
        +getRawVCache() Tensor&
    }

    class Binary_Token_Dataset {
        <<utility>>
        +uint32_t MAGIC_HEADER$
        +uint32_t CURRENT_VERSION$
        +save(file_path, tokens)$ bool
        +load(file_path, tokens)$ bool
        +existsAndValid(file_path)$ bool
    }

    class Bpe_Tokenizer {
        -unordered_map~string, int32_t~ vocab_
        -vector~string~ id_to_token_
        -unordered_map~uint64_t, Merge_Rule~ merges_
        -unordered_set~int32_t~ special_tokens_
        -int32_t bos_id_
        -int32_t eos_id_
        -int32_t unk_id_
        +load(tokenizer_json_path) bool
        +encode(text, add_bos) vector~int32_t~
        +decode(token_ids, skip_special) string
        +getVocabSize() size_t
        +isLoaded() bool
    }

    class Syllable_Tokenizer {
        -vector~Token_Metadata~ vocab_
        -unordered_map~string, int32_t~ token_to_id_
        -int32_t bos_id_
        -int32_t eos_id_
        +load(tokenizer_json_path) bool
        +save(output_path) bool
        +trainFromText(corpus_text, min_freq, max_vocab)
        +segmentSyllables(text) vector~string~
        +encode(text, add_bos) vector~int32_t~
        +decode(token_ids, skip_special, strip_leading) string
        +getRhymingTokens(token_id, must_be_bang) vector~int32_t~
        +getVocabSize() size_t
    }

    class Vietnamese_Phonetics {
        <<utility>>
        +decomposeVietnameseChar(utf8_char, base_out, tone_out)$
        +toLowerUtf8(str)$ string
        +analyzeSyllable(word)$ Syllable_Info
        +canRhymeLucBat(a, b, require_bang)$ bool
    }

    Causal_LM o-- Causal_LM_Config : config
    Causal_LM o-- Transformer_Block : blocks
    Causal_LM o-- KV_Cache_Manager : kv_caches
    Causal_LM o-- Embedding_Layer : token_embedding
    Causal_LM o-- RMSNorm_Layer : final_norm
    Causal_LM o-- Linear_Layer : lm_head
    Causal_LM o-- Bpe_Tokenizer : tokenizer
    Causal_LM o-- Loss_Scaler : loss_scaler
    Syllable_Tokenizer ..> Vietnamese_Phonetics : phonetics analysis
```

---

## 8. Class Diagram — Population & Neuroevolution (Tiến hóa Nơ-ron)

```mermaid
classDiagram
    class Population {
        -size_t population_size
        -size_t state_dimension
        -size_t action_space_size
        -Execution_Target execution_target
        -mt19937 random_engine
        -bool is_mixed_precision_enabled
        -vector~Population_Layer_Adapter~ layer_adapters
        -Tensor batched_input_tensor
        -vector~float~ batched_input_host
        +Population(template_network, pop_size)
        +evolve(fitness_scores, elitism_ratio, mutation_rate, mutation_strength)
        +selectBatchActions(inputs) vector~size_t~
        +forwardBatch(batched_inputs) Tensor
        +getIndividual(index) Neural_Network
        +setIndividual(index, network)
        +getBestIndividual(fitness_scores) Neural_Network
        +saveCheckpoint(path, fitness_scores)
        +loadCheckpoint(path)
        +setMixedPrecision(enable)
        +setExecutionTarget(target)
    }

    class Population_Layer_Adapter {
        +ILayer* template_layer
        +vector~Shape~ param_shapes
        +vector~bool~ param_evolvable
        +vector~size_t~ param_numel
        +vector~Tensor~ batched_params
        +vector2D~float~ host_params
    }

    Population o-- Population_Layer_Adapter : layer_adapters
    Population ..> Neural_Network : constructs on-demand
    Population_Layer_Adapter o-- Tensor : batched_params
```

---

## 9. Sequence Diagram — JIT Operator Fusion Pipeline (Ghép Toán tử Đồ thị)

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Tensor
    participant EE as Execution_Engine
    participant CG as Compute_Graph
    participant GO as Graph_Optimizer
    participant GX as Graph_Executor
    participant PCM as Pipeline_Cache_Manager
    participant SG as Shader_Generator
    participant VK as Vulkan_Context

    User->>Tensor: mat_a + mat_b (Elementwise ADD)
    Tensor->>CG: addNode(ADD_compute_node)
    User->>Tensor: result.relu() (Elementwise RELU)
    Tensor->>CG: addNode(RELU_compute_node)

    User->>EE: executeGraph()
    EE->>EE: computeGraphSignature(current_graph)

    alt Graph Template NOT in cache (Cache Miss)
        EE->>GO: buildCachedTemplate(current_graph)
        GO->>GO: isFusible(producer, consumer)
        Note over GO: Checks: Consumer is ELEMENTWISE,<br/>No cooperative matrix in consumer,<br/>No shared memory or multi-write hazard
        GO->>GO: hasCompatibleDimensions(ADD, RELU)
        Note over GO: Checks: Workgroups match OR<br/>producer threads at least consumer threads
        GO->>GO: hasAliasingHazard(ADD, RELU)
        Note over GO: Checks: No RAW or WAR buffer aliasing
        Note over GO: All validation checks PASS, Fused Node created!
        GO-->>EE: Cached_Graph_Template [Fused_Node(ADD + RELU)]

        EE->>GX: generateFusedGlsl(fused_node)
        GX->>SG: generateGlsl(fused_operations)
        SG-->>GX: Combined GLSL compute shader string
        GX-->>EE: fused_glsl_code

        EE->>PCM: getOrCreatePipeline(fused_glsl_code)
        PCM->>PCM: compileGlslToSpirv()
        PCM->>VK: vkCreateComputePipeline()
        PCM-->>EE: VkPipeline (cached and ready)
    else Graph Template in cache (Cache Hit)
        EE->>GO: applyCachedTemplateInPlace(raw, template, cached_graph)
    end

    EE->>GX: executeNode(fused_node, cmd_buffer)
    GX->>GX: isBuffersMatching(descriptor_entry)
    GX->>VK: vkCmdBindPipeline(fused_pipeline)
    GX->>VK: vkCmdBindDescriptorSets(shared_buffers)
    GX->>VK: vkCmdPushConstants(aligned_push_constants)
    GX->>VK: vkCmdDispatch(workgroup_x, y, z)
    EE->>VK: submitCompute(cmd_buffer)
    EE-->>User: Execution complete, GPU results ready
```

---

## 10. Sequence Diagram — LLM Autoregressive Generation (Sinh Token & KV Cache)

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant LLM as Causal_LM
    participant Tok as Bpe_Tokenizer
    participant EMB as Embedding_Layer
    participant Block as Transformer_Block (N Layers)
    participant KVC as KV_Cache_Manager (N Layers)
    participant NORM as Final RMSNorm_Layer
    participant HEAD as LM Head (Linear_Layer)
    participant EE as Execution_Engine

    User->>LLM: generate("Trăm năm", max_new_tokens=64, temp=0.7)
    LLM->>Tok: encode("Trăm năm", add_bos=true)
    Tok-->>LLM: prompt_token_ids [1, 154, 829]

    loop For each token step (up to max_new_tokens)
        LLM->>LLM: forward(current_tokens, use_cache=true)
        LLM->>EMB: forward(input_indices)
        EMB-->>LLM: hidden_states (B, S, hidden_dim)

        loop For each layer block (l = 0..N-1)
            LLM->>Block: forward(hidden_states, kv_caches[l])
            Block->>Block: input_layernorm.forward(hidden_states)
            Block->>Block: q_proj, k_proj, v_proj
            Block->>Block: applyRoPE(Q, K)
            Block->>KVC: append(K, V)
            Block->>KVC: getK(), getV()
            KVC-->>Block: full_cached_K, full_cached_V
            Block->>Block: FlashAttention(Q, cached_K, cached_V, is_causal=true)
            Block->>Block: o_proj + residual add
            Block->>Block: post_attention_layernorm.forward()
            Block->>Block: gate_proj and up_proj through swiglu to down_proj
            Block->>Block: residual add
            Block-->>LLM: layer_output
        end

        LLM->>NORM: forward(layer_output)
        LLM->>HEAD: forward(normed_output)
        HEAD-->>LLM: logits (B, S, vocab_size)

        opt GPU Execution Target
            LLM->>EE: executeGraph()
        end

        LLM->>LLM: apply repetition_penalty
        LLM->>LLM: sampleToken(logits[last_token], temp, top_p, top_k)
        LLM-->>LLM: next_token_id

        alt next_token_id == eos_token_id OR matches stop_sequences
            Note over LLM: Termination condition met, break generation loop
        else Continues generation
            LLM->>Tok: decode([next_token_id], skip_special=true)
            Tok-->>LLM: next_token_string
            opt token_callback registered
                LLM->>User: token_callback(next_token_string) [Streaming]
            end
            LLM->>LLM: append next_token_id to current_tokens
        end
    end

    LLM-->>User: complete generated string
```

---

## 11. State Diagram — Execution Engine Graph Lifecycle (Vòng đời Đồ thị Tính toán)

```mermaid
stateDiagram-v2
    [*] --> Idle : Engine Initialized

    Idle --> Recording : Tensor operator dispatched
    Recording --> Recording : Append Compute_Node to graph

    Recording --> CheckingCache : Execute or warm cache
    
    state CheckingCache <<choice>>
    CheckingCache --> ApplyingCache : Cache Hit on signature
    CheckingCache --> Optimizing : Cache Miss on signature

    state Optimizing {
        [*] --> AnalyzingHazards : Iterate nodes
        AnalyzingHazards --> ValidatingFusion : Check isFusible and dimensions
        ValidatingFusion --> CreatingFusedNode : Validation passed and buffer shared
        ValidatingFusion --> StartingNewNode : Validation failed or limit reached
        CreatingFusedNode --> AnalyzingHazards : Process next node
        StartingNewNode --> AnalyzingHazards : Process next node
        AnalyzingHazards --> [*] : All nodes partitioned
    }

    Optimizing --> GeneratingShader : Template constructed
    GeneratingShader --> CompilingSpirv : Generate fused GLSL code
    CompilingSpirv --> CachingPipeline : Compile SPIR-V and create pipeline
    CachingPipeline --> ApplyingCache : Store in pipeline cache

    ApplyingCache --> BindingResources : Assign active buffer descriptors
    
    state BindingResources {
        [*] --> CheckingDescriptors : Check isBuffersMatching
        CheckingDescriptors --> UpdatingDescriptors : Buffers changed update sets
        CheckingDescriptors --> UsingCachedDescriptors : Buffers unchanged reuse sets
        UpdatingDescriptors --> [*]
        UsingCachedDescriptors --> [*]
    }

    BindingResources --> RecordingCommands : Bind pipeline descriptors and push constants
    RecordingCommands --> Dispatching : Dispatch compute workgroups
    Dispatching --> Submitting : End command buffer
    Submitting --> AwaitingGpu : Queue submit with timeline fence
    AwaitingGpu --> CleaningGarbage : Fence signaled frame complete
    CleaningGarbage --> Idle : Clean staging buffers and clear graph
```
