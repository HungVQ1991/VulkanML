# VulkanML — Sơ đồ Kiến trúc UML Chi tiết

## 1. Architecture Layering Diagram

```mermaid
graph TD
    subgraph L6["Layer 6 — Application & High-Level Models"]
        NN["Neural_Network\n(Sequential Model + Training Loop)"]
        POP["Population\n(Neuroevolution + Batched GPU GEMM)"]
        TC["Training_Context\n(Layer Factory + Optimizer Factory)"]
        LLM["Causal_LM\n(Decoder-only Transformer LLM)"]
        RL["DQN / PPO Agents\n(Replay Buffer + CartPole Env)"]
    end

    subgraph L5["Layer 5 — Tokenizers"]
        BPE["Bpe_Tokenizer\n(Byte-Pair Encoding)"]
        SYL["Syllable_Tokenizer\n(Vietnamese Syllable)"]
        VNP["Vietnamese_Phonetics\n(Tone + Rhyme Analysis)"]
    end

    subgraph L4["Layer 4 — Concrete Neural Layers"]
        LIN["Linear_Layer\n(Dense + Adam Fused)"]
        CONV["Conv2d_Layer"]
        BN["BatchNorm 1D/2D"]
        RMSN["RMSNorm_Layer"]
        SWIGLU["SwiGLU_Layer"]
        EMB["Embedding_Layer"]
        TRB["Transformer_Block\n(QKV + FlashAttention + FFN)"]
        ACT["ReLU / GELU / Softmax"]
        POOL["MaxPool2D / GlobalAvgPool2D"]
        RES["ResNet Block / ResNet-20"]
        PPO_L["PPO_Actor_Critic_Layer"]
    end

    subgraph L4i["Layer 4 — Loss / Optimizer / LR"]
        COST["ICost_Function\n(MSE / MAE / BCE / CCE / Huber / Fused_CCE)"]
        OPT["IOptimizer\n(Adam / SGD)"]
        LRS["ILearning_Rate\n(Cosine / Step / Plateau / Exp / Poly)"]
        IL["ILayer\n(Abstract Base)"]
    end

    subgraph L3["Layer 3 — Tensor & Math System"]
        TENS["Tensor\n(Public API)"]
        SHP["Shape + Stride"]
        CPU_IMPL["Cpu_Tensor_Impl\n(~2800 lines CPU ops)"]
        GPU_IMPL["Gpu_Tensor_Impl\n(~3100 lines GPU dispatch)"]
        TIMPL["Tensor_Impl\n(Abstract Interface)"]
    end

    subgraph L2["Layer 2 — Graph & Execution Engine"]
        EE["Execution_Engine\n(Singleton, Graph Lifecycle)"]
        GO["Graph_Optimizer\n(JIT Fusion + Aliasing Hazard)"]
        GX["Graph_Executor\n(Vulkan Dispatch + Static Buffer)"]
        CG["Compute_Graph\n(DAG of Compute_Node)"]
        LS["Loss_Scaler\n(FP16 AMP)"]
        ADP["Async_Data_Pipeline\n(Double Buffer)"]
    end

    subgraph L1["Layer 1 — Vulkan Core Engine"]
        VK["Vulkan_Context\n(Device + Queue + Swapchain)"]
        NET["Vulkan_Network\n(Pipeline Layout + Descriptor Pool)"]
        ALLOC["Vulkan_Sub_Allocator\n(Memory Pool)"]
        GPUV["gpu::vector\n(GPU Buffer RAII)"]
        SHAD["Shader_Dictionary\n(JSON Metadata)"]
        SHGEN["Shader_Generator\n(Runtime GLSL)"]
        SHCOMP["Shader_Compiler\n(glslc / shaderc)"]
        PCM["Pipeline_Cache_Manager\n(On-disk VkPipeline Cache)"]
    end

    subgraph L0["Layer 0 — Utilities & Helpers"]
        LOG["Logger\n(Feature-masked Async)"]
        UPREFS["User_Preferences\n(JSON Config)"]
        PROF["Training_Profiler\n(Step Timings)"]
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

## 2. Class Diagram — Tensor System

```mermaid
classDiagram
    class Shape {
        -vector~size_t~ dims
        +getRank() size_t
        +getTotalElements() size_t
        +computeContiguousStrides() Stride
        +toString() string
        +operator[]() size_t
    }

    class Tensor_Impl {
        <<abstract>>
        #Shape shape
        #Stride strides
        #size_t total_elements
        #Data_Type data_type
        +getShape() Shape
        +getRows() size_t
        +getColumns() size_t
        +isContiguous() bool
        +validateSameDimensions()
        +matmul()* = 0
        +add()* = 0
        +relu()* = 0
        +flashAttentionForward()* = 0
        +fusedCrossEntropyLoss()* = 0
        +adamUpdate()* = 0
    }

    class Cpu_Tensor_Impl {
        -vector~float~ data
        +getData() vector~float~
        +uploadData()
        +matmul()
        +add()
        +relu()
        +conv2d()
        +flashAttentionForward()
        +fusedCrossEntropyLoss()
        +adamUpdate()
    }

    class Gpu_Tensor_Impl {
        -shared_ptr~gpu::vector~ gpu_vec
        +getVector() shared_ptr~gpu::vector~
        +uploadData()
        +matmul()
        +add()
        +relu()
        +conv2d()
        +flashAttentionForward()
        +fusedCrossEntropyLoss()
        +adamUpdate()
    }

    class Tensor {
        -shared_ptr~Tensor_Impl~ impl
        +Tensor(rows, cols, target)
        +Tensor(rows, cols, data, target)
        +matmul(other) Tensor
        +operator+() Tensor
        +relu() Tensor
        +to(Data_Type) Tensor
        +saveToFile()
        +loadFromFile()
        +getRows() size_t
        +getColumns() size_t
        +getShape() Shape
        +getData() vector~float~
        +getStorage() Storage_Handle
    }

    class gpu_vector["gpu::vector"] {
        -VkBuffer buffer
        -VkDeviceMemory memory
        -size_t size_bytes
        +getBuffer() VkBuffer
        +getSize() size_t
        +upload(data)
        +download() vector~float~
    }

    Tensor_Impl <|-- Cpu_Tensor_Impl
    Tensor_Impl <|-- Gpu_Tensor_Impl
    Tensor o-- Tensor_Impl : impl (PIMPL)
    Gpu_Tensor_Impl o-- gpu_vector : gpu_vec
    Shape <-- Tensor_Impl : shape
```

---

## 3. Class Diagram — Layer System

```mermaid
classDiagram
    class ILayer {
        <<abstract>>
        #bool is_accumulated
        #bool is_mixed_precision_enabled
        +forward(input) Tensor
        +backward(grad) Tensor*= 0
        +clone() unique_ptr~ILayer~*= 0
        +saveInference(ofstream)*= 0
        +loadInference(ifstream)*= 0
        +getLayerType() Layer_Type*= 0
        +getExecutionTarget() Execution_Target*= 0
        +setExecutionTarget()*= 0
        +getParametersAndGradients()
        +hasParameters() bool
        +setMixedPrecision(bool)
        +logBufferAddress()
    }

    class Linear_Layer {
        -Tensor weights
        -Tensor biases
        -Tensor weights_gradient
        -Tensor bias_gradient
        -Tensor input_cache
        +forward(input) Tensor
        +backward(grad) Tensor
        +saveInference()
        +loadInference()
        +getLayerType() LINEAR
    }

    class Conv2d_Layer {
        -Tensor weights
        -Tensor biases
        -uint32_t kernel_size
        -uint32_t stride, padding
        +forward(input) Tensor
        +backward(grad) Tensor
    }

    class Transformer_Block {
        -Linear_Layer q_proj, k_proj, v_proj, o_proj
        -RMSNorm_Layer attn_norm, ffn_norm
        -SwiGLU_Layer ffn
        -size_t num_heads, head_dim
        +forward(input) Tensor
        +backward(grad) Tensor
    }

    class RMSNorm_Layer {
        -Tensor gamma
        -Tensor inv_rms_cache
        +forward(input) Tensor
        +backward(grad) Tensor
    }

    class SwiGLU_Layer {
        -Linear_Layer gate_proj
        -Linear_Layer up_proj
        -Linear_Layer down_proj
        +forward(input) Tensor
        +backward(grad) Tensor
    }

    class Embedding_Layer {
        -Tensor weights
        -Tensor indices_cache
        +forward(indices) Tensor
        +backward(grad) Tensor
    }

    class Batch_Norm_Layer {
        -Tensor gamma, beta
        -Tensor running_mean, running_variance
        +forward(input) Tensor
        +backward(grad) Tensor
    }

    class Res_Net_Block_2d_Layer {
        -vector~unique_ptr~ILayer~~ main_path
        -Conv2d_Layer skip_conv
        +forward(input) Tensor
        +backward(grad) Tensor
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
    Transformer_Block o-- RMSNorm_Layer
    Transformer_Block o-- SwiGLU_Layer
    Transformer_Block o-- Linear_Layer
    Res_Net_Block_2d_Layer o-- Conv2d_Layer
```

---

## 4. Class Diagram — Execution Engine & Graph Optimization

```mermaid
classDiagram
    class Execution_Engine {
        <<singleton>>
        -unique_ptr~Vulkan_Context~ context
        -unique_ptr~Vulkan_Network~ network
        -unique_ptr~Graph_Executor~ graph_executor
        -unique_ptr~Graph_Optimizer~ graph_optimizer
        -unique_ptr~Shader_Dictionary~ shader_dictionary
        -unique_ptr~Pipeline_Cache_Manager~ pipeline_cache_manager
        -Compute_Graph current_graph
        +getInstance() Execution_Engine&
        +executeGraph()
        +buildCachedTemplate() Cached_Graph_Template
        +applyCachedTemplate()
        +getCurrentGraph() Compute_Graph&
        +isCooperativeMatrixSupported() bool
        +setCooperativeMatrixEnabled(bool)
    }

    class Compute_Graph {
        -vector~Compute_Node~ nodes
        +addNode(Compute_Node)
        +getNodeCount() size_t
        +getNodes() vector~Compute_Node~
        +clear()
    }

    class Compute_Node {
        +Compute_Pipeline pipeline_id
        +vector~shared_ptr~gpu::vector~~ buffers
        +vector~uint8_t~ push_constants_data
        +uint32_t workgroup_count_x, y, z
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
        <<static>>
        -MAX_PUSH_CONSTANTS_BYTES 128
        -MAX_STORAGE_BUFFER_BINDINGS 32
        -MAX_FUSED_OPERATIONS 8
        +optimize(Compute_Graph&)
        +buildCachedTemplate(Compute_Graph&) Cached_Graph_Template
        +applyCachedTemplate()
        -isFusible() bool
        -hasCompatibleDimensions() bool
        -hasAliasingHazard() bool
        -optimizeInternal() Cached_Graph_Template
    }

    class Graph_Executor {
        -Vulkan_Context context
        -Vulkan_Network network
        -Pipeline_Cache_Manager pipeline_cache_manager
        -Shader_Dictionary shader_dictionary
        +executeNode(Compute_Node, VkCommandBuffer)
        +generateFusedGlsl(Compute_Node) string
        +getExternalBufferIndices()
        +executeFallbackNode()
        +isBuffersMatching() bool
    }

    class Cached_Graph_Template {
        +vector~Compute_Node~ fused_nodes
        +vector~vector~Buffer_Binding_Mapping~~ buffer_mappings
        +vector~vector~Push_Constant_Mapping~~ push_constants_mappings
        +array~Compute_Graph, 2~ instantiated_graphs
        +bool is_valid
        +isValid() bool
    }

    class Pipeline_Cache_Manager {
        -Vulkan_Context context
        -VkPipelineLayout pipeline_layout
        -map~string, VkPipeline~ pipeline_cache
        +getOrCreatePipeline(glsl_code) VkPipeline
        +initializePipelineCache(path)
        +savePipelineCache(path)
    }

    Execution_Engine o-- Compute_Graph
    Execution_Engine o-- Graph_Executor
    Execution_Engine o-- Pipeline_Cache_Manager
    Compute_Graph o-- Compute_Node : "1..*"
    Compute_Node o-- Fused_Operation : "0..*"
    Graph_Optimizer ..> Cached_Graph_Template : builds
    Graph_Optimizer ..> Compute_Node : validates & fuses
    Graph_Executor ..> Compute_Node : dispatches
```

---

## 5. Class Diagram — Vulkan Core Engine

```mermaid
classDiagram
    class Vulkan_Context {
        -VkInstance instance
        -VkPhysicalDevice physical_device
        -VkDevice device
        -VkQueue compute_queue
        -VkCommandPool command_pool
        -bool cooperative_matrix_supported
        +getDevice() VkDevice
        +getQueue() VkQueue
        +allocateBuffer(size, usage, props) VkBuffer
        +submitCompute(VkCommandBuffer)
        +prepareFrame()
        +presentFrame()
        +isCooperativeMatrixEnabled() bool
    }

    class Vulkan_Sub_Allocator {
        -vector~Memory_Chunk~ chunks
        -Memory_Planner planner
        +allocate(size_bytes) Memory_Allocation
        +free(Memory_Allocation)
        +collectGarbage()
        +planStaticLayout(tensors)
    }

    class Memory_Chunk {
        -VkDeviceMemory memory
        -VkBuffer buffer
        -size_t capacity
        -vector~Free_Block~ free_blocks
    }

    class gpu_vector["gpu::vector"] {
        -VkBuffer buffer
        -size_t size_elements
        -bool owns_memory
        +getBuffer() VkBuffer
        +getSize() size_t
        +upload(vector~float~)
        +download() vector~float~
        +resize(new_size)
        +isEmpty() bool
    }

    class Vulkan_Network {
        -Vulkan_Context context
        -VkPipelineLayout pipeline_layout
        -VkDescriptorSetLayout descriptor_set_layout
        -map~Compute_Pipeline, VkPipeline~ pipelines
        +getPipeline(id) VkPipeline
        +getPipelineLayout() VkPipelineLayout
        +createPipelineFromSpirv(spv)
    }

    class Shader_Dictionary {
        <<singleton>>
        -map~Compute_Pipeline, Snippet_Metadata~ metadata_map
        +getMetadata(pipeline_id) Snippet_Metadata
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
        +addSpecializationConstant(id, name, type, value)
        +generateGlsl(ops) string
    }

    class Shader_Compiler {
        +compileGlslToSpirv(glsl_code) vector~uint32_t~
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

    Vulkan_Context <-- Vulkan_Sub_Allocator
    Vulkan_Context <-- Vulkan_Network
    Vulkan_Sub_Allocator o-- Memory_Chunk : "1..*"
    Vulkan_Network o-- Shader_Dictionary
    Shader_Dictionary o-- Snippet_Metadata : "1..*"
    Shader_Generator ..> Shader_Compiler : feeds GLSL
    Shader_Compiler ..> Vulkan_Network : SPIR-V pipeline
```

---

## 6. Class Diagram — Neural Network & Training Context

```mermaid
classDiagram
    class Neural_Network {
        -vector~unique_ptr~ILayer~~ layers
        -unique_ptr~IOptimizer~ optimizer
        -unique_ptr~ICost_Function~ cost_function
        -unique_ptr~ILearning_Rate~ learning_rate
        -unique_ptr~Loss_Scaler~ loss_scaler
        -Execution_Target execution_target
        -bool is_training_mode
        +addLayer()
        +forward(input) Tensor
        +backward(grad) Tensor
        +trainStep(input, target) float
        +fit(dataset, epochs)
        +compileAndWarmup(batch, in_dim, out_dim)
        +saveInference(path)
        +loadInference(path)
        +saveTrainingCheckpoint(path)
        +loadTrainingCheckpoint(path)
        +setMixedPrecision(bool)
    }

    class Training_Context {
        -unique_ptr~IOptimizer~ optimizer
        -unique_ptr~ILearning_Rate~ learning_rate
        -unique_ptr~ICost_Function~ cost_function
        +constructLayerFromConfig(type, stream) ILayer*
        +createCostFunction(type) ICost_Function*
        +createOptimizer(type) IOptimizer*
        +createLearningRateScheduler(type) ILearning_Rate*
    }

    class IOptimizer {
        <<abstract>>
        +step(param_grad_pairs)*= 0
        +saveCheckpoint(ofstream)*= 0
        +loadCheckpoint(ifstream)*= 0
        +getLearningRate() float*= 0
        +getType() Optimizer_Type*= 0
        +setLearningRate(float)*= 0
    }

    class Adam_Optimizer {
        -float learning_rate
        -float beta1, beta2, epsilon
        -float weight_decay
        -size_t timestep
        -map~Tensor*, Parameter_State~ states
        +step(param_grad_pairs)
        +saveCheckpoint()
        +loadCheckpoint()
    }

    class Sgd_Optimizer {
        -float learning_rate
        -float max_gradient
        +step(param_grad_pairs)
    }

    class ILearning_Rate {
        <<abstract>>
        +updateRate() float*= 0
        +step(current_value)*= 0
        +getCurrentRate() float*= 0
        +saveCheckpoint()*= 0
    }

    class ICost_Function {
        <<abstract>>
        +computeLoss(pred, target) float*= 0
        +computeGradient(pred, target) Tensor*= 0
        +isFused() bool
        +supportsIntegerTargets() bool
        +getType() Cost_Type*= 0
    }

    class Fused_Cross_Entropy {
        -int32_t ignore_index
        +computeLoss(pred, target) float
        +computeGradient(pred, target) Tensor
        +computeLossAndGradient(logits, integer_targets, d_logits) float
        +isFused() true
        +supportsIntegerTargets() true
    }

    Neural_Network o-- ILayer : "0..*"
    Neural_Network o-- IOptimizer
    Neural_Network o-- ICost_Function
    Neural_Network o-- ILearning_Rate
    Neural_Network o-- Loss_Scaler
    Training_Context ..> ILayer : constructs
    Training_Context ..> IOptimizer : constructs
    Training_Context ..> ICost_Function : constructs
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

## 7. Class Diagram — LLM & Tokenizer Subsystem

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
        +Execution_Target execution_target
        +Data_Type data_type
        +bool use_loss_scaler
        +string tokenizer_path
    }

    class Causal_LM {
        -Causal_LM_Config config
        -Embedding_Layer token_embedding
        -vector~unique_ptr~Transformer_Block~~ blocks
        -vector~unique_ptr~KV_Cache_Manager~~ kv_caches
        -RMSNorm_Layer final_norm
        -Linear_Layer lm_head
        -Bpe_Tokenizer tokenizer
        -unique_ptr~Loss_Scaler~ loss_scaler
        -unique_ptr~ICost_Function~ cost_function
        +forward(token_ids, use_cache) Tensor
        +backward(d_logits) Tensor
        +trainStep(inputs, targets, optimizer) float
        +forwardLossAndBackward(inputs, targets) float
        +generate(prompt, max_tokens, temperature, top_p, top_k) string
        +stepOptimizer(optimizer, max_grad_norm)
        +resetKVCaches()
        +saveCheckpoint(path)
        +loadInference(path)
    }

    class KV_Cache_Manager {
        -Tensor key_cache
        -Tensor value_cache
        -size_t current_seq_len
        -size_t max_seq_len
        -size_t num_heads
        -size_t head_dim
        +appendKV(keys, values)
        +getKeys() Tensor&
        +getValues() Tensor&
        +getCurrentSeqLen() size_t
        +reset()
    }

    class Binary_Token_Dataset {
        -string file_path
        -size_t seq_len
        -size_t stride
        -size_t sample_count
        +getSampleCount() size_t
        +getBatch(start, batch_size) pair~vector~vector~int~~, vector~vector~int~~~
        +createFromText(txt_path, bin_path, tokenizer, seq_len, stride)
    }

    class Bpe_Tokenizer {
        -unordered_map~string, int32_t~ vocab
        -vector~Merge_Rule~ merges
        +encode(text) vector~int32_t~
        +decode(ids) string
        +load(path)
        +save(path)
        +getVocabSize() size_t
    }

    class Syllable_Tokenizer {
        -vector~Token_Metadata~ vocab
        -unordered_map~string, int32_t~ token_to_id
        +encode(text) vector~int32_t~
        +decode(ids) string
        +load(path)
        +getVocabSize() size_t
    }

    Causal_LM o-- Causal_LM_Config
    Causal_LM o-- Transformer_Block : "N blocks"
    Causal_LM o-- KV_Cache_Manager : "N caches"
    Causal_LM o-- Embedding_Layer
    Causal_LM o-- RMSNorm_Layer
    Causal_LM o-- Linear_Layer : lm_head
    Causal_LM o-- Bpe_Tokenizer
    Causal_LM o-- Loss_Scaler
    Bpe_Tokenizer ..> Syllable_Tokenizer : alternative
```

---

## 8. Class Diagram — Population (Neuroevolution)

```mermaid
classDiagram
    class Population {
        -vector~Neural_Network~ individuals
        -size_t pop_size
        -vector~vector~Tensor~~ population_weights_soa
        -float elitism_ratio
        -float mutation_rate
        -float mutation_strength
        +evolve(fitness_scores)
        +selectBatchActions(inputs) vector~size_t~
        +getIndividual(index) Neural_Network&
        +setIndividual(index, network)
        +saveCheckpoint(path, fitness)
        +loadCheckpoint(path)
        +getBestIndividual(fitness) Neural_Network&
        +setMixedPrecision(bool)
    }

    class Population_Layer_Adapter {
        -ILayer& layer_ref
        +getPopulationParameterDims() vector~Shape~
        +getPopulationParameter(idx) vector~float~
        +setPopulationParameter(idx, data)
        +getPopulationParameterInitializer(idx) function
    }

    Population o-- Neural_Network : "pop_size individuals"
    Population o-- Population_Layer_Adapter
```

---

## 9. Sequence Diagram — Operator Fusion Pipeline (JIT)

```mermaid
sequenceDiagram
    participant User
    participant Tensor
    participant EE as Execution_Engine
    participant CG as Compute_Graph
    participant GO as Graph_Optimizer
    participant GX as Graph_Executor
    participant PCM as Pipeline_Cache_Manager
    participant SG as Shader_Generator
    participant VK as Vulkan_Context

    User->>Tensor: mat_a + mat_b (ADD node)
    Tensor->>CG: addNode(ADD_compute_node)
    User->>Tensor: result.relu() (RELU node)
    Tensor->>CG: addNode(RELU_compute_node)

    User->>EE: executeGraph()
    EE->>GO: buildCachedTemplate(current_graph)

    GO->>GO: isFusible(ELEMENTWISE, ELEMENTWISE)?
    GO->>GO: hasCompatibleDimensions(ADD, RELU)?
    GO->>GO: hasAliasingHazard(ADD, RELU)?
    Note over GO: All checks PASS → Fuse!

    GO-->>EE: Cached_Graph_Template {fused_node[ADD+RELU]}

    EE->>PCM: getOrCreatePipeline(fused_glsl_code)
    alt Pipeline NOT in cache
        PCM->>SG: generateFusedGlsl(fused_operations)
        SG-->>PCM: "void main() { add(); relu(); }"
        PCM->>PCM: compileGlslToSpirv()
        PCM->>VK: vkCreateComputePipeline()
        PCM-->>EE: VkPipeline (new)
    else Pipeline IN cache
        PCM-->>EE: VkPipeline (cached)
    end

    EE->>GX: executeNode(fused_node, cmd_buffer)
    GX->>VK: vkCmdBindPipeline(fused_pipeline)
    GX->>VK: vkCmdBindDescriptorSets(buffers)
    GX->>VK: vkCmdPushConstants(ADD_params + RELU_params)
    GX->>VK: vkCmdDispatch(workgroups)
    EE->>VK: vkQueueSubmit()
    EE-->>User: Result tensors ready
```

---

## 10. Sequence Diagram — LLM Token Generation (Autoregressive + KV Cache)

```mermaid
sequenceDiagram
    participant User
    participant LLM as Causal_LM
    participant Tok as Bpe_Tokenizer
    participant EMB as Embedding_Layer
    participant Block as Transformer_Block (x N)
    participant KVC as KV_Cache_Manager
    participant NORM as Final RMSNorm
    participant HEAD as LM Head (Linear)
    participant SAMPLE as Token Sampler

    User->>LLM: generate("Trăm năm", max_tokens=64, temperature=0.7)
    LLM->>Tok: encode("Trăm năm")
    Tok-->>LLM: [token_ids: 154, 829]

    loop For each new token (up to max_tokens)
        LLM->>EMB: forward(token_ids)
        EMB-->>LLM: hidden_states (seq_len × hidden_dim)

        loop For each Transformer Block
            Block->>Block: Pre-RMSNorm(hidden_states)
            Block->>Block: Q, K, V = qkv_proj(normed)
            Block->>Block: applyRoPE(Q, K)
            Block->>KVC: appendKV(K, V)
            KVC-->>Block: full_keys, full_values (prefix + current)
            Block->>Block: FlashAttention(Q, full_K, full_V, causal=true)
            Block->>Block: O = o_proj(attn_output)
            Block->>Block: hidden = hidden + O (residual)
            Block->>Block: Pre-RMSNorm(hidden)
            Block->>Block: FFN = SwiGLU(normed)
            Block->>Block: hidden = hidden + FFN (residual)
        end

        LLM->>NORM: rmsNormForward(hidden_states)
        LLM->>HEAD: forward(normed) → logits (vocab_size)
        LLM->>SAMPLE: sampleToken(logits, temperature, top_p, top_k)
        SAMPLE-->>LLM: next_token_id

        alt next_token_id == eos_token
            LLM-->>User: decoded_text (stop)
        else
            LLM->>Tok: decode([next_token_id])
            Tok-->>LLM: next_token_text
            LLM-->>User: stream callback(next_token_text)
        end
    end
```

---

## 11. State Diagram — Execution Engine Graph Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Idle : Engine Initialized

    Idle --> Recording : Tensor Op Triggered
    Recording --> Recording : addNode() to Compute_Graph
    Recording --> Optimizing : executeGraph() called

    Optimizing --> FusingNodes : Graph_Optimizer.buildCachedTemplate()
    FusingNodes --> FusingNodes : isFusible? hasCompatibleDimensions? hasAliasingHazard?
    FusingNodes --> CompilingPipeline : Fusion decisions finalized

    CompilingPipeline --> PipelineCached : getOrCreatePipeline()
    CompilingPipeline --> FallbackMode : Shader compilation FAILED
    PipelineCached --> Dispatching : Pipeline ready

    Dispatching --> Dispatching : vkCmdBindPipeline + vkCmdDispatch per node
    Dispatching --> Submitted : vkQueueSubmit()
    Submitted --> Waiting : Timeline Semaphore / Fence
    Waiting --> Idle : GPU work complete, graph cleared

    FallbackMode --> Dispatching : Per-node fallback execution
```
