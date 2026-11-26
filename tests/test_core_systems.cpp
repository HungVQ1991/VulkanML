#include "test_common.h"

bool testSgdOptimizer(Execution_Target exec_target)
{
    Tensor param_mat(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor grad(2, 2, { 0.5f, -1.0f, 2.0f, -3.0f }, exec_target);

    Sgd_Optimizer optimizer(0.1f, 100.0f);
    optimizer.step({ { &param_mat, &grad } });

    return verifyMatrix(param_mat, { 0.95f, 2.1f, 2.8f, 4.3f });
}

bool testAdamOptimizer(Execution_Target exec_target)
{
    Tensor param_mat(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor grad_mat(1, 2, { 0.1f, -0.2f }, exec_target);

    Adam_Optimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 100.0f);
    optimizer.step({ { &param_mat, &grad_mat } });

    if (!verifyMatrix(param_mat, { 0.999f, 2.001f }))
    {
        return false;
    }

    Tensor param_wd(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor grad_wd(1, 2, { 0.1f, -0.2f }, exec_target);

    Adam_Optimizer optimizer_wd(0.001f, 0.9f, 0.999f, 1e-8f, 100.0f, 0.1f);
    optimizer_wd.step({ { &param_wd, &grad_wd } });

    return verifyMatrix(param_wd, { 0.9989f, 2.0008f });
}

bool testLearningRateSchedulers()
{
    No_Decay no_decay(0.01f);
    no_decay.step();
    bool no_decay_ok = nearlyEqual(no_decay.getCurrentRate(), 0.01f);

    Step_Decay step_decay(0.1f, 1e-6f, 0.5f, 2);
    step_decay.step();
    bool step1_ok = nearlyEqual(step_decay.getCurrentRate(), 0.1f);
    step_decay.step();
    bool step2_ok = nearlyEqual(step_decay.getCurrentRate(), 0.05f);

    Multi_Step_Decay multi_step(0.1f, 1e-6f, 0.1f, { 2.0f, 4.0f });
    multi_step.step();
    multi_step.step();
    bool multi_step_ok = nearlyEqual(multi_step.getCurrentRate(), 0.01f);

    Exponential_Decay exp_decay(0.1f, 1e-6f, 0.9f);
    exp_decay.step();
    bool exp_ok = nearlyEqual(exp_decay.getCurrentRate(), 0.09f);

    Cosine_Annealing cosine_decay(0.1f, 0.0f, 4);
    cosine_decay.step();
    cosine_decay.step();
    bool cosine_ok = nearlyEqual(cosine_decay.getCurrentRate(), 0.05f);

    Polynomial_Decay poly_decay(0.1f, 0.0f, 4);
    poly_decay.step();
    poly_decay.step();
    bool poly_ok = nearlyEqual(poly_decay.getCurrentRate(), 0.05f);

    Reduce_On_Plateau plateau_decay(0.1f, 1e-6f, 0.5f, 2, false);
    plateau_decay.step(1.0f);
    plateau_decay.step(1.2f);
    plateau_decay.step(1.3f);
    bool plateau_ok = nearlyEqual(plateau_decay.getCurrentRate(), 0.05f);

    Cosine_Annealing warmup_cosine(0.1f, 0.0f, 10, 2);
    warmup_cosine.step();
    bool warmup1_ok = nearlyEqual(warmup_cosine.getCurrentRate(), 0.05f);
    warmup_cosine.step();
    bool warmup2_ok = nearlyEqual(warmup_cosine.getCurrentRate(), 0.1f);
    warmup_cosine.step();
    bool warmup3_ok = (warmup_cosine.getCurrentRate() < 0.1f) && (warmup_cosine.getCurrentRate() > 0.0f);

    return no_decay_ok && step1_ok && step2_ok && multi_step_ok && exp_ok && cosine_ok && poly_ok && plateau_ok && warmup1_ok && warmup2_ok && warmup3_ok;
}

bool testMatrixSerialization(Execution_Target exec_target)
{
    const std::string temp_file = makeUniqueTestTempPath("temp_matrix_serialization").string();
    Scoped_Temp_File_Guard temp_guard({temp_file});
    Tensor original(2, 3, { 1.0f, -2.5f, 3.2f, 4.8f, 5.0f, -6.1f }, exec_target);

    std::ofstream out_file(temp_file, std::ios::binary);
    if (!out_file.is_open())
    {
        return false;
    }
    original.saveTensor(out_file);
    out_file.close();

    std::ifstream in_file(temp_file, std::ios::binary);
    if (!in_file.is_open())
    {
        return false;
    }
    Tensor loaded = Tensor::loadTensor(in_file, exec_target);
    in_file.close();

    return verifyMatrix(loaded, { 1.0f, -2.5f, 3.2f, 4.8f, 5.0f, -6.1f });
}

bool testModelInferenceSerialization(Execution_Target exec_target)
{
    const std::string temp_file = makeUniqueTestTempPath("temp_model_inference").string();
    Scoped_Temp_File_Guard temp_guard({temp_file});

    Neural_Network network(exec_target);
    network.addLayer<Linear_Layer>(2, 3, exec_target);
    network.addLayer<Relu_Layer>(exec_target);
    network.addLayer<Linear_Layer>(3, 1, exec_target);

    Tensor input_data(1, 2, { 1.5f, -0.5f }, exec_target);
    Tensor pred_before = network.forward(input_data);

    network.saveInference(temp_file);

    Neural_Network loaded_network(exec_target);
    loaded_network.loadInference(temp_file, exec_target);

    Tensor pred_after = loaded_network.forward(input_data);

    return verifyMatrix(pred_after, pred_before.getData());
}

bool testGpuVectorLifecycle()
{
    Execution_Engine& engine = Execution_Engine::getInstance();
    const Vulkan_Context& context = engine.getContext();

    size_t initial_count = 1024;
    auto vec = std::make_unique<gpu::vector>(context, initial_count);
    if (vec->getSize() != initial_count || vec->getBuffer() == VK_NULL_HANDLE)
    {
        return false;
    }

    std::vector<float> upload_sample(initial_count, 3.1415f);
    vec->uploadData(upload_sample);

    std::vector<float> download_sample;
    vec->downloadData(download_sample);
    if (download_sample.size() != initial_count || !nearlyEqual(download_sample[0], 3.1415f))
    {
        return false;
    }

    size_t expanded_count = 2048;
    vec->allocateMemory(expanded_count);
    if (vec->getSize() != expanded_count)
    {
        return false;
    }

    gpu::vector moved_vec = std::move(*vec);
    if (moved_vec.getSize() != expanded_count || vec->getBuffer() != VK_NULL_HANDLE)
    {
        return false;
    }

    return true;
}

bool testVulkanSubAllocatorAndGarbageCollection()
{
    Execution_Engine& engine = Execution_Engine::getInstance();
    const Vulkan_Context& context = engine.getContext();
    Vulkan_Sub_Allocator& allocator = context.getAllocator();

    VkMemoryRequirements mem_req{
        .size = 1024 * 1024,
        .alignment = 256,
        .memoryTypeBits = 0xFFFFFFFF };

    Memory_Allocation alloc_a = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Memory_Allocation alloc_b = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (alloc_a.memory == VK_NULL_HANDLE || alloc_b.memory == VK_NULL_HANDLE)
    {
        return false;
    }

    VkDeviceSize offset_a = alloc_a.offset;
    allocator.free(alloc_a);

    Memory_Allocation alloc_c = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bool reuse_ok = (alloc_c.offset == offset_a);

    allocator.free(alloc_b);
    allocator.free(alloc_c);

    uint32_t current_frame = context.getCurrentFrame();
    Memory_Allocation garbage_alloc = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    context.deferDestruction(current_frame, VK_NULL_HANDLE, garbage_alloc);

    context.cleanGarbage(current_frame);

    return reuse_ok;
}

bool testOperatorFusionAndGraphExecution()
{
    Execution_Engine& engine = Execution_Engine::getInstance();
    engine.getCurrentGraph().clear();

    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_b(2, 2, { 2.0f, 3.0f, 4.0f, 5.0f }, Execution_Target::VULKAN_GPU);

    Tensor mat_add = mat_a + mat_b;
    Tensor mat_relu = mat_add.relu();

    size_t raw_node_count = engine.getCurrentGraph().getNodeCount();
    if (raw_node_count < 2)
    {
        return false;
    }

    engine.executeGraph();
    if (!verifyMatrix(mat_relu, { 3.0f, 5.0f, 7.0f, 9.0f }))
    {
        return false;
    }

    Tensor mat_c(2, 2, { -10.0f, 5.0f, 0.0f, 2.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_d(2, 2, { 3.0f, 2.0f, 1.0f, -5.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_add2 = mat_c + mat_d;
    Tensor mat_relu2 = mat_add2.relu();

    engine.executeGraph();
    if (!verifyMatrix(mat_relu2, { 0.0f, 7.0f, 1.0f, 0.0f }))
    {
        return false;
    }

    Tensor mat_e(2, 2, { 10.0f, -20.0f, 30.0f, -40.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_f(2, 2, { 5.0f, 5.0f, -5.0f, -5.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_add3 = mat_e + mat_f;
    Tensor mat_relu3 = mat_add3.relu();

    engine.executeGraph();
    return verifyMatrix(mat_relu3, { 15.0f, 0.0f, 25.0f, 0.0f });
}

bool testBatchedTensorMatmulFusion()
{
    Execution_Engine& engine = Execution_Engine::getInstance();
    engine.getCurrentGraph().clear();

    Tensor t_a(Shape{ 2, 2, 2 }, {
        -5.0f, 2.0f,
        3.0f, -4.0f,
        1.0f, -2.0f,
        -3.0f, 4.0f
        }, Execution_Target::VULKAN_GPU);

    Tensor t_w(Shape{ 2, 2 }, {
        1.0f, 0.0f,
        0.0f, 1.0f
        }, Execution_Target::VULKAN_GPU);

    Tensor t_b(Shape{ 1, 2 }, { 1.0f, 0.0f }, Execution_Target::VULKAN_GPU);

    Tensor t_mmadd = t_a.matmulAdd(t_w, t_b);
    Tensor t_fused = t_mmadd.relu();

    engine.executeGraph();

    return verifyMatrix(t_fused, { 0.0f, 2.0f, 4.0f, 0.0f, 2.0f, 0.0f, 0.0f, 4.0f });
}

class Dummy_Data_Pipeline : public Async_Data_Pipeline
{
protected:
    void prepareBatchHost(size_t, std::vector<float>& output_inputs, std::vector<float>& output_targets) override
    {
        std::fill(output_inputs.begin(), output_inputs.end(), 1.0f);
        std::fill(output_targets.begin(), output_targets.end(), 2.0f);
    }

public:
    Dummy_Data_Pipeline() : Async_Data_Pipeline() {}
    ~Dummy_Data_Pipeline() override { stop(); }
    size_t getBatchSize() const override { return 2; }
};

bool testAsyncDataPipeline()
{
    Dummy_Data_Pipeline pipeline;
    pipeline.initializeBuffers(2, 2, 2, Execution_Target::CPU);
    pipeline.start();

    Batch_Data batch1 = pipeline.nextBatch(2, 2, 2);
    bool b1_ok = (batch1.input_matrix != nullptr && batch1.input_matrix->getData().size() == 4);

    Batch_Data batch2 = pipeline.nextBatch(2, 2, 2);
    bool b2_ok = (batch2.target_matrix != nullptr && batch2.target_matrix->getData().size() == 4);

    pipeline.stop();
    return b1_ok && b2_ok;
}

bool testReplayBuffer()
{
    constexpr size_t capacity = 4;
    Replay_Buffer buffer(capacity, 1337);

    if (buffer.getSize() != 0 || buffer.getCapacity() != capacity || buffer.isReady(1))
    {
        return false;
    }

    for (size_t i = 0; i < 5; ++i)
    {
        buffer.push(Transition{
            .state = { static_cast<float>(i), static_cast<float>(i + 1) },
            .action = i % 2,
            .reward = static_cast<float>(i) * 0.5f,
            .next_state = { static_cast<float>(i + 1), static_cast<float>(i + 2) },
            .is_terminal = (i == 4) });
    }

    if (buffer.getSize() != capacity || !buffer.isReady(capacity) || buffer.isReady(capacity + 1))
    {
        return false;
    }

    Transition_Batch batch = buffer.sample(2);
    bool batch_dim_ok = (batch.batch_size == 2 && batch.state_dimension == 2 &&
        batch.states.size() == 4 && batch.next_states.size() == 4 &&
        batch.actions.size() == 2 && batch.rewards.size() == 2 &&
        batch.terminals.size() == 2);

    buffer.clear();
    bool clear_ok = (buffer.getSize() == 0 && !buffer.isReady(1));

    return batch_dim_ok && clear_ok;
}

bool testDqnAgent(Execution_Target exec_target)
{
    constexpr size_t state_dim = 2;
    constexpr size_t action_dim = 2;

    Neural_Network q_net(exec_target);
    q_net.addLayer<Linear_Layer>(state_dim, action_dim, exec_target);
    q_net.setCostFunction<Huber_Cost>(1.0f, exec_target);
    q_net.setOptimizer<Adam_Optimizer>(0.01f);

    Neural_Network target_net(exec_target);
    target_net.addLayer<Linear_Layer>(state_dim, action_dim, exec_target);
    target_net.setCostFunction<Huber_Cost>(1.0f, exec_target);
    target_net.setOptimizer<Adam_Optimizer>(0.01f);

    Dqn_Agent agent(std::move(q_net), state_dim, action_dim, 100, 0.99f, 0.0f, 0.01f, 0.95f, exec_target, 1234);
    agent.initializeTargetNetworkFromPrototype(std::move(target_net));
    agent.setTargetUpdateParameters(2, false);

    size_t deterministic_action = agent.selectAction({ 1.0f, 0.5f }, false);
    bool action_ok = (deterministic_action < action_dim);

    for (size_t i = 0; i < 6; ++i)
    {
        agent.storeTransition(Transition{
            .state = { static_cast<float>(i), 1.0f },
            .action = i % action_dim,
            .reward = 1.0f,
            .next_state = { static_cast<float>(i + 1), 1.0f },
            .is_terminal = (i % 3 == 0) });
    }

    agent.trainStep(4);
    agent.trainStep(4);

    agent.setEpsilon(1.0f);
    agent.decayEpsilon();
    bool epsilon_ok = nearlyEqual(agent.getEpsilon(), 0.95f);

    agent.synchronizeTargetNetworkSoft(0.1f);
    agent.synchronizeTargetNetworkHard();

    return action_ok && epsilon_ok;
}

bool testPopulation(Execution_Target exec_target)
{
    constexpr size_t pop_size = 8;
    constexpr size_t state_dim = 16;
    constexpr size_t action_dim = 4;
    constexpr size_t hidden_dim = 32;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, hidden_dim, exec_target);
    template_net.addLayer<Gelu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(hidden_dim, action_dim, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target, 42);
    if (pop.getPopulationSize() != pop_size || pop.getStateDimension() != state_dim || pop.getActionSpaceSize() != action_dim)
    {
        return false;
    }

    Population pop_tmpl(pop_size, template_net, state_dim, action_dim, exec_target, 123);
    if (pop_tmpl.getPopulationSize() != pop_size)
    {
        return false;
    }

    std::vector<float> test_state(state_dim);
    for (size_t i = 0; i < state_dim; ++i)
    {
        test_state[i] = std::sin(static_cast<float>(i) * 0.5f);
    }

    for (size_t i = 0; i < pop_size; ++i)
    {
        size_t act1 = pop.selectAction(i, test_state);
        size_t act2 = pop.selectAction(i, test_state);
        if (act1 != act2 || act1 >= action_dim)
        {
            return false;
        }
    }

    Neural_Network ind0 = pop.getIndividual(0);
    Tensor ind0_input(1, state_dim, test_state, exec_target);
    Tensor ind0_output = ind0.forward(ind0_input);
    flushIfGpu(exec_target);
    std::vector<float> q_vals = ind0_output.getData();
    size_t expected_act0 = 0;
    float max_q = q_vals[0];
    for (size_t a = 1; a < q_vals.size(); ++a)
    {
        if (q_vals[a] > max_q)
        {
            max_q = q_vals[a];
            expected_act0 = a;
        }
    }
    if (pop.selectAction(0, test_state) != expected_act0)
    {
        return false;
    }

    std::vector<float> flat_states(pop_size * state_dim);
    std::vector<size_t> expected_batch(pop_size);
    for (size_t i = 0; i < pop_size; ++i)
    {
        for (size_t d = 0; d < state_dim; ++d)
        {
            flat_states[i * state_dim + d] = std::cos(static_cast<float>(i * state_dim + d) * 0.2f);
        }
        expected_batch[i] = pop.selectAction(i, flat_states.data() + (i * state_dim));
    }

    std::vector<size_t> batch_results(pop_size);
    pop.selectBatchActions(flat_states.data(), nullptr, pop_size, batch_results.data());
    for (size_t i = 0; i < pop_size; ++i)
    {
        if (batch_results[i] != expected_batch[i])
        {
            return false;
        }
    }

    std::vector<size_t> active_indices = { 1, 3, 6 };
    std::vector<float> subset_states(active_indices.size() * state_dim);
    for (size_t k = 0; k < active_indices.size(); ++k)
    {
        size_t idx = active_indices[k];
        std::copy(flat_states.begin() + idx * state_dim,
            flat_states.begin() + (idx + 1) * state_dim,
            subset_states.begin() + k * state_dim);
    }
    std::vector<size_t> subset_results(active_indices.size());
    pop.selectBatchActions(subset_states.data(), active_indices.data(), active_indices.size(), subset_results.data());
    for (size_t k = 0; k < active_indices.size(); ++k)
    {
        if (subset_results[k] != expected_batch[active_indices[k]])
        {
            return false;
        }
    }

    const std::string temp_file = makeUniqueTestTempPath("temp_population_individual").string();
    Scoped_Temp_File_Guard temp_guard({temp_file});
    if (!pop.saveIndividual(2, temp_file))
    {
        return false;
    }

    Population pop_loader(pop_size, template_net, state_dim, action_dim, exec_target, 9999);
    if (!pop_loader.loadIndividual(5, temp_file))
    {
        return false;
    }

    for (size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop_loader.selectAction(5, st) != pop.selectAction(2, st))
        {
            return false;
        }
    }

    constexpr size_t elite_candidate = 4;
    std::vector<size_t> ind4_actions(pop_size);
    for (size_t i = 0; i < pop_size; ++i)
    {
        ind4_actions[i] = pop.selectAction(elite_candidate, flat_states.data() + i * state_dim);
    }

    std::vector<float> fitness(pop_size, 0.0f);
    fitness[elite_candidate] = 1000.0f;

    Neural_Network best_ind_before = pop.getBestIndividual(fitness.data());
    Tensor in_matrix_best(1, state_dim, std::vector<float>(flat_states.begin(), flat_states.begin() + state_dim), exec_target);
    Tensor out_matrix_best = best_ind_before.forward(in_matrix_best);
    flushIfGpu(exec_target);
    const auto& best_raw_data = out_matrix_best.getData();
    size_t best_initial_act = static_cast<size_t>(std::distance(best_raw_data.begin(), std::max_element(best_raw_data.begin(), best_raw_data.end())));
    if (best_initial_act != ind4_actions[0])
    {
        return false;
    }

    const std::string best_temp = makeUniqueTestTempPath("temp_best_champ").string();
    Scoped_Temp_File_Guard best_guard({best_temp});
    if (!pop.saveBestIndividual(best_temp, fitness.data()))
    {
        return false;
    }
    if (!pop_loader.loadBestIndividual(best_temp))
    {
        return false;
    }

    for (size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop_loader.selectAction(0, st) != ind4_actions[i])
        {
            return false;
        }
    }

    const std::string checkpoint_temp = makeUniqueTestTempPath("temp_population_checkpoint").string();
    Scoped_Temp_File_Guard ckpt_temp_guard({checkpoint_temp});
    uint64_t saved_gen = 105;
    if (!pop.saveCheckpoint(checkpoint_temp, saved_gen, fitness.data()))
    {
        return false;
    }

    uint64_t loaded_gen = 0;
    std::vector<float> loaded_fitness(pop_size, 0.0f);
    if (!pop_loader.loadCheckpoint(checkpoint_temp, loaded_gen, loaded_fitness.data()))
    {
        return false;
    }
    if (loaded_gen != saved_gen)
    {
        return false;
    }
    for (size_t i = 0; i < pop_size; ++i)
    {
        if (!nearlyEqual(loaded_fitness[i], fitness[i]))
        {
            return false;
        }
    }

    uint64_t loaded_gen_null = 0;
    if (!pop_loader.loadCheckpoint(checkpoint_temp, loaded_gen_null, nullptr))
    {
        return false;
    }
    if (loaded_gen_null != saved_gen)
    {
        return false;
    }

    pop.evolve(fitness.data(), 0.15f, 0.0f, 0.0f, 0.0f, 3, 1);
    for (size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop.selectAction(0, st) != ind4_actions[i])
        {
            return false;
        }
    }

    pop.evolve(fitness.data(), 0.15f, 0.8f, 0.3f, 0.5f, 3, 1);

    return true;
}

bool testLayerInterfaceContracts(Execution_Target exec_target)
{
    std::mt19937 random_engine(1337);

    Linear_Layer lin(2, 3, exec_target, 2.0f);
    auto lin_clone = lin.clone();
    if (!lin_clone || lin_clone->getLayerType() != Layer_Type::LINEAR || lin_clone->getExecutionTarget() != exec_target)
    {
        return false;
    }
    auto lin_dims = lin.getPopulationParameterDims();
    if (lin_dims.size() != 2 || lin_dims[0] != Shape{ 2, 3 } || lin_dims[1] != Shape{ 1, 3 })
    {
        return false;
    }
    auto lin_evolvable = lin.getPopulationParameterIsEvolvable();
    if (lin_evolvable.size() != 2 || !lin_evolvable[0] || !lin_evolvable[1])
    {
        return false;
    }
    auto lin_init_w = lin.getPopulationParameterInitializer(0);
    auto lin_init_b = lin.getPopulationParameterInitializer(1);
    if (!lin_init_w || !lin_init_b || lin_init_b(random_engine) != 0.0f)
    {
        return false;
    }
    lin_clone->setPopulationParameter(0, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    lin_clone->setPopulationParameter(1, { 0.1f, 0.2f, 0.3f });
    Tensor lin_in(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor lin_out = lin_clone->forward(lin_in);
    if (!verifyMatrix(lin_out, { 9.1f, 12.2f, 15.3f }, (exec_target == Execution_Target::VULKAN_GPU) ? 0.02f : 1e-3f))
    {
        return false;
    }
    bool lin_bounds_ok = false;
    try
    {
        lin.setPopulationParameter(2, {});
    }
    catch (const std::out_of_range&)
    {
        lin_bounds_ok = true;
    }
    if (!lin_bounds_ok)
    {
        return false;
    }

    Relu_Layer relu(exec_target);
    auto relu_clone = relu.clone();
    if (!relu_clone || relu_clone->getLayerType() != Layer_Type::RELU || !relu.getPopulationParameterDims().empty() || !relu.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }
    bool relu_throw_ok = false;
    try
    {
        relu.setPopulationParameter(0, { 1.0f });
    }
    catch (const std::out_of_range&)
    {
        relu_throw_ok = true;
    }
    if (!relu_throw_ok)
    {
        return false;
    }

    Gelu_Layer gelu(exec_target);
    auto gelu_clone = gelu.clone();
    if (!gelu_clone || gelu_clone->getLayerType() != Layer_Type::GELU || !gelu.getPopulationParameterDims().empty() || !gelu.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    Softmax_Layer softmax(false, exec_target);
    auto softmax_clone = softmax.clone();
    if (!softmax_clone || softmax_clone->getLayerType() != Layer_Type::SOFTMAX || !softmax.getPopulationParameterDims().empty() || !softmax.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    Batch_Norm_Layer bn1d(4, 1e-5f, 0.1f, exec_target);
    auto bn1d_clone = bn1d.clone();
    if (!bn1d_clone || bn1d.getPopulationParameterDims().size() != 4 || bn1d.getPopulationParameterIsEvolvable() != std::vector<bool>{true, true, false, false})
    {
        return false;
    }
    auto bn1d_init_gamma = bn1d.getPopulationParameterInitializer(0);
    auto bn1d_init_beta = bn1d.getPopulationParameterInitializer(1);
    auto bn1d_init_mean = bn1d.getPopulationParameterInitializer(2);
    auto bn1d_init_var = bn1d.getPopulationParameterInitializer(3);
    if (bn1d_init_gamma(random_engine) != 1.0f || bn1d_init_beta(random_engine) != 0.0f || bn1d_init_mean(random_engine) != 0.0f || bn1d_init_var(random_engine) != 1.0f)
    {
        return false;
    }

    Batch_Norm_2d_Layer bn2d(2, 2, 3, 1e-5f, 0.1f, exec_target);
    auto bn2d_clone = bn2d.clone();
    if (!bn2d_clone || bn2d.getPopulationParameterDims().size() != 4 || bn2d.getPopulationParameterIsEvolvable() != std::vector<bool>{true, true, false, false})
    {
        return false;
    }

    Conv2d_Layer conv(4, 4, 1, 2, 3, 1, 1, exec_target);
    auto conv_clone = conv.clone();
    if (!conv_clone || conv.getPopulationParameterDims().size() != 2 || conv.getPopulationParameterIsEvolvable() != std::vector<bool>{true, true})
    {
        return false;
    }
    if (conv.getPopulationParameterDims()[0] != Shape{ 3, 3, 1, 2 } || conv.getPopulationParameterDims()[1] != Shape{ 1, 2 })
    {
        return false;
    }

    Max_Pool_2d_Layer maxpool(4, 4, 2, 2, 2, 0, exec_target);
    auto maxpool_clone = maxpool.clone();
    if (!maxpool_clone || !maxpool.getPopulationParameterDims().empty() || !maxpool.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    Global_Avg_Pool_2d_Layer gap(4, 4, 2, exec_target);
    auto gap_clone = gap.clone();
    if (!gap_clone || !gap.getPopulationParameterDims().empty() || !gap.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    PPO_Actor_Critic_Layer ppo(2, exec_target);
    ppo.addActorLayer<Linear_Layer>(2, 2, exec_target);
    ppo.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    auto ppo_clone = ppo.clone();
    if (!ppo_clone || ppo.getPopulationParameterDims().size() != 4 || ppo.getPopulationParameterIsEvolvable().size() != 4)
    {
        return false;
    }

    Res_Net_Block_2d_Layer res_block(4, 4, 8, 8, 1, exec_target);
    auto res_clone = res_block.clone();
    if (!res_clone || res_block.getPopulationParameterDims().empty() || res_block.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    return true;
}

bool testPopulationDeepNetwork(Execution_Target exec_target)
{
    constexpr size_t pop_size = 6;
    constexpr size_t state_dim = 8;
    constexpr size_t action_dim = 3;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, 16, exec_target);
    template_net.addLayer<Relu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(16, 8, exec_target);
    template_net.addLayer<Gelu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(8, action_dim, exec_target);
    template_net.addLayer<Softmax_Layer>(false, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target, 888);
    if (pop.getPopulationSize() != pop_size || pop.getStateDimension() != state_dim || pop.getActionSpaceSize() != action_dim)
    {
        std::cout << "Place 1";
        return false;
    }

    std::vector<float> input_sample(state_dim);
    for (size_t i = 0; i < state_dim; ++i)
    {
        input_sample[i] = 0.1f * static_cast<float>(i + 1);
    }

    for (size_t i = 0; i < pop_size; ++i)
    {
        size_t act = pop.selectAction(i, input_sample);
        if (act >= action_dim)
        {
            std::cout << "Place 2";
            return false;
        }

        Neural_Network ind = pop.getIndividual(i);
        Tensor in_mat(1, state_dim, input_sample, exec_target);
        Tensor out_mat = ind.forward(in_mat);
        flushIfGpu(exec_target);
        const auto& out_data = out_mat.getData();
        size_t expected_act = static_cast<size_t>(std::distance(out_data.begin(), std::max_element(out_data.begin(), out_data.end())));
        if (act != expected_act)
        {
            std::cout << "Place 3";
            return false;
        }
    }

    std::vector<float> batch_input(pop_size * state_dim);
    for (size_t i = 0; i < pop_size * state_dim; ++i)
    {
        batch_input[i] = std::sin(static_cast<float>(i) * 0.3f);
    }

    std::vector<size_t> batch_actions(pop_size);
    pop.selectBatchActions(batch_input.data(), nullptr, pop_size, batch_actions.data());

    for (size_t i = 0; i < pop_size; ++i)
    {
        size_t single_act = pop.selectAction(i, batch_input.data() + i * state_dim);
        if (batch_actions[i] != single_act)
        {
            std::cout << "Place 4";
            return false;
        }
    }

    std::vector<float> fitness(pop_size);
    for (size_t i = 0; i < pop_size; ++i)
    {
        fitness[i] = static_cast<float>(i * 10);
    }

    Neural_Network best_ind = pop.getBestIndividual(fitness.data());
    Tensor in_mat_best(1, state_dim, input_sample, exec_target);
    Tensor best_out = best_ind.forward(in_mat_best);
    flushIfGpu(exec_target);
    const auto& best_data = best_out.getData();
    size_t best_expected = static_cast<size_t>(std::distance(best_data.begin(), std::max_element(best_data.begin(), best_data.end())));
    if (pop.selectAction(pop_size - 1, input_sample) != best_expected)
    {
        std::cout << "Place 5";
        return false;
    }

    const std::string ckpt_path = makeUniqueTestTempPath("temp_deep_pop").string();
    Scoped_Temp_File_Guard ckpt_guard({ckpt_path});
    uint64_t gen_in = 50;
    if (!pop.saveCheckpoint(ckpt_path, gen_in, fitness.data()))
    {
        std::cout << "Place 6";
        return false;
    }

    Population pop_loader(pop_size, template_net, state_dim, action_dim, exec_target, 111);
    uint64_t gen_out = 0;
    std::vector<float> fit_out(pop_size);
    if (!pop_loader.loadCheckpoint(ckpt_path, gen_out, fit_out.data()))
    {
        std::cout << "Place 7";
        return false;
    }

    if (gen_out != gen_in)
    {
        std::cout << "Place 8";
        return false;
    }
    for (size_t i = 0; i < pop_size; ++i)
    {
        if (!nearlyEqual(fit_out[i], fitness[i]))
        {
            std::cout << "Place 9";
            return false;
        }
        if (pop_loader.selectAction(i, input_sample) != pop.selectAction(i, input_sample))
        {
            std::cout << "Place 9";
            return false;
        }
    }
    return true;
}

bool testMatrixConcatAndSplit(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor mat_b(2, 1, { 5.0f, 6.0f }, exec_target);

    Tensor concat_cols_res = mat_a.concatenateColumns(mat_b);
    bool concat_cols_ok = verifyMatrix(concat_cols_res, { 1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f });

    auto [split_left, split_right] = concat_cols_res.splitColumns(2);
    bool split_cols_ok = verifyMatrix(split_left, { 1.0f, 2.0f, 3.0f, 4.0f }) &&
        verifyMatrix(split_right, { 5.0f, 6.0f });

    Tensor mat_row_a(1, 2, { 10.0f, 20.0f }, exec_target);
    Tensor mat_row_b(2, 2, { 30.0f, 40.0f, 50.0f, 60.0f }, exec_target);

    Tensor concat_rows_res = mat_row_a.concatenateRows(mat_row_b);
    bool concat_rows_ok = verifyMatrix(concat_rows_res, { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f });

    auto [split_up, split_down] = concat_rows_res.splitRows(1);
    bool split_rows_ok = verifyMatrix(split_up, { 10.0f, 20.0f }) &&
        verifyMatrix(split_down, { 30.0f, 40.0f, 50.0f, 60.0f });

    return concat_cols_ok && split_cols_ok && concat_rows_ok && split_rows_ok;
}

bool testPpoActorCriticForward(Execution_Target exec_target)
{
    constexpr uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto& actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Tensor(2, 2, { 1.0f, 0.0f, 0.0f, 1.0f }, exec_target));
    actor_linear.setBiases(Tensor(1, 2, { 0.5f, -0.5f }, exec_target));

    auto& critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Tensor(2, 1, { 1.0f, 2.0f }, exec_target));
    critic_linear.setBiases(Tensor(1, 1, { 1.0f }, exec_target));

    Tensor input_matrix(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor output_matrix = ppo_layer.forward(input_matrix);

    bool output_ok = verifyMatrix(output_matrix, { 1.5f, 1.5f, 6.0f, 3.5f, 3.5f, 12.0f });
    output_matrix.print();
    bool actor_sub_ok = verifyMatrix(ppo_layer.getActorOutput(), { 1.5f, 1.5f, 3.5f, 3.5f });
    ppo_layer.getActorOutput().print();
    bool critic_sub_ok = verifyMatrix(ppo_layer.getCriticOutput(), { 6.0f, 12.0f });
    ppo_layer.getCriticOutput().print();
    return output_ok && actor_sub_ok && critic_sub_ok;
}

bool testPpoActorCriticBackward(Execution_Target exec_target)
{
    constexpr uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto& actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Tensor(2, 2, { 1.0f, 0.0f, 0.0f, 1.0f }, exec_target));
    actor_linear.setBiases(Tensor(1, 2, { 0.0f, 0.0f }, exec_target));

    auto& critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Tensor(2, 1, { 1.0f, 1.0f }, exec_target));
    critic_linear.setBiases(Tensor(1, 1, { 0.0f }, exec_target));

    Tensor input_matrix(1, 2, { 2.0f, 3.0f }, exec_target);
    ppo_layer.forward(input_matrix);

    Tensor output_gradient(1, 3, { 1.0f, 2.0f, 3.0f }, exec_target);
    Tensor input_gradient = ppo_layer.backward(output_gradient);

    bool input_grad_ok = verifyMatrix(input_gradient, { 4.0f, 5.0f });

    auto params = ppo_layer.getParametersAndGradients();
    bool params_count_ok = (params.size() == 4);

    bool actor_weight_grad_ok = verifyMatrix(*params[0].second, { 2.0f, 4.0f, 3.0f, 6.0f });
    bool actor_bias_grad_ok = verifyMatrix(*params[1].second, { 1.0f, 2.0f });
    bool critic_weight_grad_ok = verifyMatrix(*params[2].second, { 6.0f, 9.0f });
    bool critic_bias_grad_ok = verifyMatrix(*params[3].second, { 3.0f });

    return input_grad_ok && params_count_ok && actor_weight_grad_ok &&
        actor_bias_grad_ok && critic_weight_grad_ok && critic_bias_grad_ok;
}

bool testPpoActorCriticSerialization(Execution_Target exec_target)
{
    const std::string temp_file = makeUniqueTestTempPath("temp_ppo_layer_serialization").string();
    Scoped_Temp_File_Guard temp_guard({temp_file});
    constexpr uint64_t action_dim = 2;

    PPO_Actor_Critic_Layer ppo_source(action_dim, exec_target);
    auto& actor_linear = ppo_source.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Tensor(2, 2, { 1.5f, -0.5f, 0.5f, 2.0f }, exec_target));
    actor_linear.setBiases(Tensor(1, 2, { 0.1f, -0.2f }, exec_target));

    auto& critic_linear = ppo_source.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Tensor(2, 1, { 0.8f, -1.2f }, exec_target));
    critic_linear.setBiases(Tensor(1, 1, { 0.5f }, exec_target));

    Tensor input_mat(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor pred_before = ppo_source.forward(input_mat);

    std::ofstream out_stream(temp_file, std::ios::binary);
    if (!out_stream.is_open())
    {
        return false;
    }
    ppo_source.saveInference(out_stream);
    out_stream.close();

    PPO_Actor_Critic_Layer ppo_loaded(action_dim, exec_target);
    ppo_loaded.addActorLayer<Linear_Layer>(2, 2, exec_target);
    ppo_loaded.addCriticLayer<Linear_Layer>(2, 1, exec_target);

    std::ifstream in_stream(temp_file, std::ios::binary);
    if (!in_stream.is_open())
    {
        return false;
    }
    ppo_loaded.loadInference(in_stream);
    in_stream.close();

    Tensor pred_after = ppo_loaded.forward(input_mat);

    return verifyMatrix(pred_after, pred_before.getData());
}

bool testLinearLayerInitializer(Execution_Target exec_target)
{
    Linear_Layer layer(64, 64, exec_target);
    auto init_fn = layer.getPopulationParameterInitializer(0);
    std::mt19937 rng(42);

    constexpr size_t sample_count = 1000;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < sample_count; ++i)
    {
        float val = init_fn(rng);
        sum += val;
        sum_sq += val * val;
    }
    float mean = sum / static_cast<float>(sample_count);
    float variance = (sum_sq / static_cast<float>(sample_count)) - (mean * mean);

    bool mean_ok = std::abs(mean) < 0.15f;
    bool variance_ok = variance > 0.0005f;

    auto bias_init_fn = layer.getPopulationParameterInitializer(1);
    bool bias_zero = (bias_init_fn(rng) == 0.0f);

    return mean_ok && variance_ok && bias_zero;
}

bool testReluBatchedForward(Execution_Target exec_target)
{
    Relu_Layer relu(exec_target);
    std::vector<float> input_data = { -2.0f, -0.5f, 0.0f, 1.5f, 3.0f };
    Tensor batched_input(Shape{ 1, 1, 5 }, input_data, exec_target);
    std::vector<Tensor> batched_params;

    Tensor output = relu.forward(batched_input, batched_params);
    flushIfGpu(exec_target);

    std::vector<float> expected = { 0.0f, 0.0f, 0.0f, 1.5f, 3.0f };
    std::vector<float> actual = output.getData();
    if (actual.size() != expected.size())
    {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (!nearlyEqual(actual[i], expected[i], 1e-4f))
        {
            return false;
        }
    }
    return true;
}

bool testGradientAccumulation(Execution_Target exec_target)
{
    Neural_Network net(exec_target);
    net.addLayer<Linear_Layer>(4, 8, exec_target);
    net.addLayer<Relu_Layer>(exec_target);
    net.addLayer<Linear_Layer>(8, 2, exec_target);

    std::vector<float> mb1_in_data = { 1.0f, 0.5f, -0.5f, 2.0f,
                                       -1.0f, 2.0f, 0.0f, 1.0f };
    std::vector<float> mb2_in_data = { 0.5f, -1.0f, 1.5f, 0.0f,
                                       2.0f, 1.0f, -1.0f, -0.5f };
    std::vector<float> full_in_data = mb1_in_data;
    full_in_data.insert(full_in_data.end(), mb2_in_data.begin(), mb2_in_data.end());

    std::vector<float> mb1_target_data = { 1.0f, 0.0f,
                                           0.0f, 1.0f };
    std::vector<float> mb2_target_data = { 0.5f, 0.5f,
                                           1.0f, 0.0f };
    std::vector<float> full_target_data = mb1_target_data;
    full_target_data.insert(full_target_data.end(), mb2_target_data.begin(), mb2_target_data.end());

    Tensor full_in(4, 4, full_in_data, exec_target);
    Tensor full_target(4, 2, full_target_data, exec_target);

    Tensor mb1_in(2, 4, mb1_in_data, exec_target);
    Tensor mb1_target(2, 2, mb1_target_data, exec_target);

    Tensor mb2_in(2, 4, mb2_in_data, exec_target);
    Tensor mb2_target(2, 2, mb2_target_data, exec_target);

    net.enableGradientAccumulation(false);
    net.resetGradients();
    Tensor full_out = net.forward(full_in);
    Tensor full_grad = full_out - full_target;
    net.backward(full_grad);
    flushIfGpu(exec_target);

    auto full_params_grads = net.getParametersAndGradients();
    std::vector<std::vector<float>> full_grads_data;
    for (const auto& pg : full_params_grads)
    {
        full_grads_data.push_back(pg.second->getData());
    }

    net.enableGradientAccumulation(true);
    net.resetGradients();
    flushIfGpu(exec_target);

    Tensor mb1_out = net.forward(mb1_in);
    Tensor mb1_grad = mb1_out - mb1_target;
    net.backward(mb1_grad);
    flushIfGpu(exec_target);

    std::cout << "\n--- [DEBUG: Gradient Accumulation Inspection] ---\n";
    auto mb1_params_grads = net.getParametersAndGradients();
    for (size_t i = 0; i < mb1_params_grads.size(); ++i)
    {
        std::cout << "Param [" << i << "] Gradient after Micro-batch 1 ("
            << mb1_params_grads[i].second->getRows() << "x"
            << mb1_params_grads[i].second->getColumns() << "):\n";
        mb1_params_grads[i].second->print();
    }

    Tensor mb2_out = net.forward(mb2_in);
    Tensor mb2_grad = mb2_out - mb2_target;
    net.backward(mb2_grad);
    flushIfGpu(exec_target);

    auto accum_params_grads = net.getParametersAndGradients();
    float tol = (exec_target == Execution_Target::VULKAN_GPU) ? 1e-2f : 1e-3f;
    bool all_match = true;

    for (size_t i = 0; i < accum_params_grads.size(); ++i)
    {
        std::vector<float> accum_grad = accum_params_grads[i].second->getData();
        const auto& expected_grad = full_grads_data[i];

        std::cout << "Param [" << i << "] Accumulated Grad (MB1 + MB2):\n";
        accum_params_grads[i].second->print();

        std::cout << "Param [" << i << "] Expected Full-batch Grad:\n";
        Tensor expected_tensor(accum_params_grads[i].second->getShape(), expected_grad, exec_target);
        expected_tensor.print();

        if (accum_grad.size() != expected_grad.size())
        {
            std::cout << "[FAIL] Param [" << i << "] Size mismatch: "
                << accum_grad.size() << " != " << expected_grad.size() << "\n";
            return false;
        }

        float max_diff = 0.0f;
        for (size_t j = 0; j < accum_grad.size(); ++j)
        {
            float diff = std::abs(accum_grad[j] - expected_grad[j]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            if (!nearlyEqual(accum_grad[j], expected_grad[j], tol))
            {
                all_match = false;
            }
        }
        std::cout << "Param [" << i << "] Max Abs Diff = " << max_diff << "\n\n";
    }

    if (!all_match)
    {
        return false;
    }

    net.resetGradients();
    flushIfGpu(exec_target);

    auto zeroed_grads = net.getParametersAndGradients();
    for (const auto& zg : zeroed_grads)
    {
        auto data = zg.second->getData();
        for (float v : data)
        {
            if (!nearlyEqual(v, 0.0f, 1e-5f))
            {
                return false;
            }
        }
    }

    net.enableGradientAccumulation(false);
    return true;
}

bool testCompositeLayerSerialization(Execution_Target exec_target)
{
    const std::string resnet_file = makeUniqueTestTempPath("temp_resnet_block_cfg").string();
    Scoped_Temp_File_Guard resnet_guard({resnet_file});
    {
        Res_Net_Block_2d_Layer res_block(4, 4, 8, 8, 1, exec_target);
        std::ofstream out(resnet_file, std::ios::binary);
        if (!out.is_open()) return false;
        res_block.saveConfiguration(out);
    }
    {
        std::ifstream in(resnet_file, std::ios::binary);
        if (!in.is_open()) return false;
        auto loaded_layer = Training_Context::constructLayerFromConfig(in, Layer_Type::RES_NET_BLOCK_2D, exec_target);
        if (!loaded_layer || loaded_layer->getLayerType() != Layer_Type::RES_NET_BLOCK_2D)
        {
            return false;
        }
    }

    const std::string ppo_file = makeUniqueTestTempPath("temp_ppo_block_cfg").string();
    Scoped_Temp_File_Guard ppo_guard({ppo_file});
    {
        PPO_Actor_Critic_Layer ppo(2, exec_target);
        ppo.addActorLayer<Linear_Layer>(4, 4, exec_target);
        ppo.addCriticLayer<Linear_Layer>(4, 1, exec_target);
        std::ofstream out(ppo_file, std::ios::binary);
        if (!out.is_open()) return false;
        ppo.saveConfiguration(out);
    }
    {
        std::ifstream in(ppo_file, std::ios::binary);
        if (!in.is_open()) return false;
        auto loaded_layer = Training_Context::constructLayerFromConfig(in, Layer_Type::PPO_ACTOR_CRITIC, exec_target);
        if (!loaded_layer || loaded_layer->getLayerType() != Layer_Type::PPO_ACTOR_CRITIC)
        {
            return false;
        }
    }

    return true;
}

bool testCrossCompatibilityPopulationToNetwork(Execution_Target exec_target)
{
    constexpr size_t pop_size = 4;
    constexpr size_t state_dim = 4;
    constexpr size_t action_dim = 2;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, 8, exec_target);
    template_net.addLayer<Relu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(8, action_dim, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target, 777);

    const std::string export_file = makeUniqueTestTempPath("temp_individual_inference").string();
    Scoped_Temp_File_Guard export_guard({export_file});
    if (!pop.saveIndividualAsInference(0, export_file))
    {
        return false;
    }

    Neural_Network standalone_net(exec_target);
    try
    {
        standalone_net.loadInference(export_file, exec_target);
    }
    catch (const std::exception&)
    {
        return false;
    }

    std::vector<float> sample_input = { 0.5f, -0.2f, 1.0f, -0.8f };
    Tensor in_tensor(1, state_dim, sample_input, exec_target);
    Tensor net_out = standalone_net.forward(in_tensor);
    flushIfGpu(exec_target);

    size_t pop_action = pop.selectAction(0, sample_input);
    const auto& net_data = net_out.getData();
    size_t net_action = static_cast<size_t>(std::distance(
        net_data.begin(), std::max_element(net_data.begin(), net_data.end())));

    if (pop_action != net_action)
    {
        return false;
    }

    pop.setIndividual(1, standalone_net);
    size_t pop_action_ind1 = pop.selectAction(1, sample_input);
    return (pop_action_ind1 == net_action);
}

bool testTensorInterfaceUnification(Execution_Target exec_target)
{
    const std::string temp_tensor_file = makeUniqueTestTempPath("temp_tensor_io").string();
    Scoped_Temp_File_Guard temp_guard({temp_tensor_file});
    Tensor original(Shape{ 2, 3 }, { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f }, exec_target);
    {
        std::ofstream out(temp_tensor_file, std::ios::binary);
        if (!out.is_open()) return false;
        original.saveTensor(out);
    }
    Tensor loaded;
    {
        std::ifstream in(temp_tensor_file, std::ios::binary);
        if (!in.is_open()) return false;
        loaded = Tensor::loadTensor(in, exec_target);
    }

    if (loaded.getShape() != original.getShape())
    {
        return false;
    }
    if (!verifyMatrix(loaded, original.getData()))
    {
        return false;
    }

    Linear_Layer linear(3, 2, exec_target);
    Tensor input(1, 3, { 1.0f, 0.5f, -1.0f }, exec_target);
    Tensor fwd_out = linear.forward(input);
    Tensor grad_in = linear.backward(Tensor(1, 2, { 0.1f, -0.1f }, exec_target));

    flushIfGpu(exec_target);

    return (fwd_out.getColumns() == 2 && grad_in.getColumns() == 3);
}

bool testPipelineCachePersistence()
{
    auto &engine = Execution_Engine::getInstance();
    auto &cache_mgr = engine.getPipelineCacheManager();
    if (cache_mgr.getPipelineCache() == VK_NULL_HANDLE)
    {
        return false;
    }
    cache_mgr.savePipelineCache();
    std::string path = cache_mgr.getCacheFilePath();
    return std::filesystem::exists(path);
}

bool testFp16SupportAndCastingCpu()
{
    Shape shape{2, 3};
    std::vector<float> input_data = {1.0f, -2.5f, 0.125f, 500.0f, -0.05f, 0.0f};
    Tensor fp32_tensor(shape, input_data, Execution_Target::CPU);

    Tensor fp16_tensor = fp32_tensor.toFp16();
    if (fp16_tensor.getDataType() != Data_Type::FLOAT16)
    {
        return false;
    }

    Tensor recovered_fp32 = fp16_tensor.toFp32();
    if (recovered_fp32.getDataType() != Data_Type::FLOAT32)
    {
        return false;
    }

    const auto &rec_data = recovered_fp32.getData();
    for (size_t i = 0; i < input_data.size(); ++i)
    {
        if (std::abs(rec_data[i] - input_data[i]) > 1e-2f * (std::abs(input_data[i]) + 1.0f))
        {
            return false;
        }
    }
    return true;
}

bool testFp16SupportAndCastingGpu()
{
    auto &engine = Execution_Engine::getInstance();
    const auto &context = engine.getContext();
    if (!context.isFloat16Supported() || !context.isFloat16Enabled())
    {
        std::cout << "[INFO: GPU does not enable/support FP16, skipping GPU kernel test] ";
        return true;
    }

    Shape shape{2, 3};
    std::vector<float> input_data = {1.0f, -2.5f, 0.125f, 42.0f, -100.0f, 0.5f};
    Tensor gpu_fp32(shape, input_data, Execution_Target::VULKAN_GPU);

    Tensor gpu_fp16 = gpu_fp32.toFp16();
    if (gpu_fp16.getDataType() != Data_Type::FLOAT16)
    {
        return false;
    }

    Tensor gpu_recovered_fp32 = gpu_fp16.toFp32();
    engine.waitIdle();

    const auto &rec_data = gpu_recovered_fp32.getData();
    for (size_t i = 0; i < input_data.size(); ++i)
    {
        if (std::abs(rec_data[i] - input_data[i]) > 1e-2f * (std::abs(input_data[i]) + 1.0f))
        {
            return false;
        }
    }
    return true;
}

bool testLossScalerAndAmp()
{
    Loss_Scaler scaler(1024.0f, 2.0f, 0.5f, 2, true);

    float loss = 0.05f;
    float scaled_loss = scaler.scaleLoss(loss);
    if (std::abs(scaled_loss - 51.2f) > 1e-3f)
    {
        return false;
    }

    Shape shape{2, 2};
    std::vector<float> normal_grad = {0.1f, 0.2f, -0.1f, 0.05f};
    Tensor grad_tensor(shape, normal_grad, Execution_Target::CPU);

    scaler.scaleGradient(grad_tensor);
    if (std::abs(grad_tensor.getData()[0] - 102.4f) > 1e-2f)
    {
        return false;
    }

    scaler.unscaleGradient(grad_tensor);
    if (std::abs(grad_tensor.getData()[0] - 0.1f) > 1e-3f)
    {
        return false;
    }

    if (scaler.hasOverflow(grad_tensor))
    {
        return false;
    }

    std::vector<float> overflow_grad = {0.1f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f};
    Tensor overflow_tensor(shape, overflow_grad, Execution_Target::CPU);
    if (!scaler.hasOverflow(overflow_tensor))
    {
        return false;
    }

    bool accepted = scaler.step(true);
    if (accepted || std::abs(scaler.getScaleFactor() - 512.0f) > 1e-3f)
    {
        return false;
    }

    scaler.step(false);
    scaler.step(false);
    if (std::abs(scaler.getScaleFactor() - 1024.0f) > 1e-3f)
    {
        return false;
    }

    Neural_Network nn(Execution_Target::CPU);
    nn.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);
    nn.setOptimizer<Sgd_Optimizer>(0.01f);
    nn.enableMixedPrecision(true);

    if (!nn.isMixedPrecisionEnabled())
    {
        return false;
    }

    Tensor input(1, 4, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Execution_Target::CPU);
    Tensor target(1, 2, std::vector<float>{0.5f, -0.5f}, Execution_Target::CPU);
    nn.trainStep(input, target);

    return true;
}

bool testStaticCommandBuffer()
{
    Scoped_Engine_Flags_Guard flags_guard;
    Execution_Engine &engine = Execution_Engine::getInstance();
    engine.waitIdle();

    Neural_Network nn(Execution_Target::VULKAN_GPU);
    nn.enableStaticGraph(true);
    if (!nn.isStaticGraphEnabled())
    {
        return false;
    }

    nn.addLayer<Linear_Layer>(4, 8, Execution_Target::VULKAN_GPU);
    nn.addLayer<Gelu_Layer>(Execution_Target::VULKAN_GPU);
    nn.addLayer<Linear_Layer>(8, 2, Execution_Target::VULKAN_GPU);

    nn.setOptimizer<Adam_Optimizer>(0.01f);
    nn.setCostFunction<Mse_Cost>();
    nn.setTrainingMode(true);

    if (engine.getGraphExecutor().isStaticBaked(0) || engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    uint32_t f0 = engine.getContext().getCurrentFrame();
    Tensor in1(2, 4, std::vector<float>{1.0f, 0.5f, -0.5f, 2.0f, -1.0f, 0.0f, 1.5f, -2.0f}, Execution_Target::VULKAN_GPU);
    Tensor tgt1(2, 2, std::vector<float>{0.5f, -0.5f, 1.0f, 0.0f}, Execution_Target::VULKAN_GPU);
    nn.trainStep(in1, tgt1);
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(f0))
    {
        return false;
    }

    uint32_t f1 = engine.getContext().getCurrentFrame();
    Tensor in2(2, 4, std::vector<float>{0.2f, -0.3f, 0.8f, 1.1f, -0.5f, 0.4f, -1.2f, 0.7f}, Execution_Target::VULKAN_GPU);
    Tensor tgt2(2, 2, std::vector<float>{0.1f, 0.9f, -0.3f, 0.4f}, Execution_Target::VULKAN_GPU);
    nn.trainStep(in2, tgt2);
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(f1))
    {
        return false;
    }

    if (!engine.getGraphExecutor().isStaticBaked(0) || !engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    nn.trainStep(in1, tgt1);
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(0) || !engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    nn.setTrainingMode(false);
    if (engine.getGraphExecutor().isStaticBaked(0) || engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    uint32_t f_inf = engine.getContext().getCurrentFrame();
    Tensor test_in(2, 4, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, Execution_Target::VULKAN_GPU);
    Tensor pred = nn.forward(test_in);
    engine.executeGraph();
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(f_inf))
    {
        return false;
    }

    Tensor pred2 = nn.forward(test_in);
    engine.executeGraph();
    engine.waitIdle();

    auto pred_data = pred.getData();
    if (pred_data.size() != 4)
    {
        return false;
    }

    nn.enableStaticGraph(false);
    engine.invalidateStaticGraph();
    engine.waitIdle();

    return true;
}

void runCoreTestSuite(Execution_Target exec_target, const std::string &target_name, Test_Result_Counter &counter)
{
    std::cout << "   RUNNING TEST SUITE ON " << target_name << "\n";

    std::cout << "\n[1. Basic Tensor Arithmetics]\n";
    counter.record("  Tensor Addition (with Broadcast):  ", testMatrixAddition(exec_target));
    counter.record("  Tensor Subtraction (with Broadcast): ", testMatrixSubtraction(exec_target));
    counter.record("  Tensor Multiplication (GEMM):      ", testMatrixMultiplication(exec_target));
    counter.record("  Scalar Multiplication & Division:   ", testScalarOperations(exec_target));
    counter.record("  Hadamard Multiplication & Division: ", testHadamardOperations(exec_target));
    counter.record("  Tensor Concat & Split (Row/Col):   ", testMatrixConcatAndSplit(exec_target));

    std::cout << "\n[2. Transformations & Advanced Operations]\n";
    counter.record("  Transpose & Tensor Inversion:       ", testTransposeAndInverse(exec_target));
    counter.record("  Euclidean L2 Normalization:        ", testNormalize(exec_target));
    counter.record("  Fused Linear Bias Add (MatmulAdd): ", testMatmulAdd(exec_target));
    counter.record("  Batched Tensor GEMM (3D):          ", testBatchedTensorMatmul(exec_target));
    counter.record("  Batched Tensor GEMM Add (3D):      ", testBatchedTensorMatmulAdd(exec_target));

    std::cout << "\n[3. Activation Functions]\n";
    counter.record("  ReLU Forward & Backward:           ", testRelu(exec_target));
    counter.record("  Sigmoid Forward & Backward:        ", testSigmoidLayer(exec_target));
    counter.record("  GELU Forward & Backward:           ", testGelu(exec_target));
    counter.record("  Softmax Forward & Backward:        ", testSoftmax(exec_target));

    std::cout << "\n[4. Cost & Loss Functions]\n";
    counter.record("  MSE Cost & Gradient:               ", testMseLoss(exec_target));
    counter.record("  MAE Cost & Gradient:               ", testMaeLoss(exec_target));
    counter.record("  BCE Cost & Gradient:               ", testBceLoss(exec_target));
    counter.record("  CCE Cost & Gradient:               ", testCceLoss(exec_target));
    counter.record("  Huber Cost & Gradient:             ", testHuberLoss(exec_target));
    counter.record("  ASL Cost & Gradient:               ", testAslLoss(exec_target));
    counter.record("  Fused CCE Cost & Gradient:         ", testFusedCrossEntropy(exec_target));

    std::cout << "\n[5. Neural Network Layers]\n";
    counter.record("  Linear Layer (Forward & Backward): ", testLinearLayer(exec_target));
    counter.record("  Conv2D Layer (Forward & Backward): ", testConv2dLayer(exec_target));
    counter.record("  Conv2D Native FP16 (Fwd & Bwd):    ", testConv2dLayerFp16(exec_target));
    counter.record("  Conv2D FP16 CoopMat Multi-Ch:      ", testConv2dFp16CoopMatMultiChannelParity(exec_target));
    counter.record("  Zero-Cast Pipeline (Native FP16):  ", testNativeFp16ZeroCastPipeline(exec_target));
    counter.record("  MaxPool2D Layer (with Mask):       ", testMaxPool2dLayer(exec_target));
    counter.record("  AvgPool2D Layer (Fwd & Bwd):       ", testAvgPool2dLayer(exec_target));
    counter.record("  GlobalAvgPool2D Layer:             ", testGlobalAvgPool2dLayer(exec_target));
    counter.record("  GlobalMaxPool2D Op (Fwd & Bwd):    ", testGlobalMaxPool2d(exec_target));
    counter.record("  BatchNorm 1D Layer:                ", testBatchNormLayer(exec_target));
    counter.record("  BatchNorm 2D Layer:                ", testBatchNorm2dLayer(exec_target));
    counter.record("  ResNet Block 2D (Id & Proj):       ", testResNetBlock2dLayer(exec_target));
    counter.record("  PPO Layer Forward:                 ", testPpoActorCriticForward(exec_target));
    counter.record("  PPO Layer Backward & Accumulation: ", testPpoActorCriticBackward(exec_target));
    counter.record("  PPO Layer Serialization I/O:       ", testPpoActorCriticSerialization(exec_target));
    counter.record("  RMSNorm Layer (Fwd & Bwd):         ", testRmsNormLayer(exec_target));
    counter.record("  RoPE Rotary Embedding (Fwd & Bwd): ", testRoPE(exec_target));
    counter.record("  Embedding Layer (Fwd/Bwd/Clone):   ", testEmbeddingLayer(exec_target));
    counter.record("  SwiGLU FFN Layer (Fwd & Bwd):       ", testSwiGluLayer(exec_target));

    auto safeRun = [](auto &&fn, Execution_Target target) -> bool {
        try {
            return fn(target);
        } catch (const std::exception &e) {
            std::cout << " [EXCEPTION: " << e.what() << "] ";
            return false;
        } catch (...) {
            std::cout << " [UNKNOWN EXCEPTION] ";
            return false;
        }
    };

    counter.record("  FlashAttention-2 FP16 (CoopMat):   ", safeRun(testFlashAttention, exec_target));
    counter.record("  KV Cache Manager (Paged / Ring):   ", safeRun(testKvCacheManager, exec_target));
    counter.record("  Transformer Block (Prefill & Decode): ", safeRun(testTransformerBlock, exec_target));
    counter.record("  Causal LM Model (Forward & Generate): ", safeRun(testCausalLMModel, exec_target));
    counter.record("  Causal LM Pre-training Engine Suite:  ", safeRun(testCausalLMTraining, exec_target));
    counter.record("  Causal LM Pure FP16 Engine Suite:     ", safeRun(testCausalLMFP16Training, exec_target));

    std::cout << "\n[6. Optimizers]\n";
    counter.record("  SGD Optimizer Step:                ", testSgdOptimizer(exec_target));
    counter.record("  Adam Optimizer Step:               ", testAdamOptimizer(exec_target));

    std::cout << "\n[7. Reinforcement Learning & Neuroevolution]\n";
    counter.record("  DQN Agent (Train Step & Target Sync): ", testDqnAgent(exec_target));
    counter.record("  Population Suite (GEMM/Evolve/IO): ", testPopulation(exec_target));
    counter.record("  Population Suite (Native FP16):    ", testPopulationFp16(exec_target));
    counter.record("  Population Deep Architecture:      ", testPopulationDeepNetwork(exec_target));
    counter.record("  Layer Population Interface & Clone:", testLayerInterfaceContracts(exec_target));

    std::cout << "\n[8. Serialization & I/O]\n";
    counter.record("  Tensor Binary I/O:                 ", testMatrixSerialization(exec_target));
    counter.record("  Model Inference I/O (NNI1):        ", testModelInferenceSerialization(exec_target));

    std::cout << "\n[9. Unified Architecture & New Features]\n";
    counter.record("  Linear Layer Init (He/Xavier):      ", testLinearLayerInitializer(exec_target));
    counter.record("  ReLU Batched Forward (Population):  ", testReluBatchedForward(exec_target));
    counter.record("  Gradient Accumulation (Micro-batch):", testGradientAccumulation(exec_target));
    counter.record("  Composite Layer Deserialization:    ", testCompositeLayerSerialization(exec_target));
    counter.record("  Pop to NN Inference Compatibility:  ", testCrossCompatibilityPopulationToNetwork(exec_target));
    counter.record("  Tensor Interface Unification:       ", testTensorInterfaceUnification(exec_target));
    std::cout << "\n";
}

void runEngineSystemsSuite(Test_Result_Counter &counter)
{
    counter.record("  Learning Rate Schedulers Suite:    ", testLearningRateSchedulers());
    counter.record("  GPU Vector Lifecycle & Resizing:   ", testGpuVectorLifecycle());
    counter.record("  Sub-Allocator & Garbage Collector: ", testVulkanSubAllocatorAndGarbageCollection());
    counter.record("  Operator Fusion & Graph Dispatch:  ", testOperatorFusionAndGraphExecution());
    counter.record("  Batched GEMM + ReLU Fusion:        ", testBatchedTensorMatmulFusion());
    counter.record("  Async Data Pipeline Double-Buffer: ", testAsyncDataPipeline());
    counter.record("  Replay Buffer Capacity & Sampling: ", testReplayBuffer());
    counter.record("  Pipeline Cache Persistence:        ", testPipelineCachePersistence());
    counter.record("  FP16 CPU Support & Precision Cast: ", testFp16SupportAndCastingCpu());
    counter.record("  FP16 GPU Support & Precision Cast: ", testFp16SupportAndCastingGpu());
    counter.record("  Loss Scaler & Mixed Precision AMP: ", testLossScalerAndAmp());
    counter.record("  Static Command Buffer (Replay):    ", testStaticCommandBuffer());
    counter.record("  BPE Tokenizer (UTF-8 & Fallback):  ", testBpeTokenizer());
}
