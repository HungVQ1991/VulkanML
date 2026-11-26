#include "layer/lora_linear_layer.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

LoRA_Linear_Layer::LoRA_Linear_Layer() = default;

LoRA_Linear_Layer::LoRA_Linear_Layer(size_t _input_dimension,
                                     size_t _output_dimension,
                                     size_t _rank,
                                     float _alpha,
                                     float _dropout_rate,
                                     Execution_Target _execution_target,
                                     float _initialization_gain,
                                     Data_Type _data_type,
                                     bool _use_bias)
    : base_layer(_input_dimension, _output_dimension, _execution_target, _initialization_gain, _data_type, _use_bias),
      cache_scaled_delta(_execution_target),
      cache_casted_delta(_execution_target),
      cache_output(_execution_target),
      cache_dy_scaled(_execution_target),
      cache_dx_lora(_execution_target),
      cache_dx_total(_execution_target)
{
    setMixedPrecision(base_layer.isMixedPrecisionEnabled());
    if (_rank > 0)
    {
        enableLoRA(_rank, _alpha, _dropout_rate, true);
    }
}

LoRA_Linear_Layer::LoRA_Linear_Layer(Linear_Layer _base_layer,
                                     size_t _rank,
                                     float _alpha,
                                     float _dropout_rate)
    : base_layer(std::move(_base_layer)),
      cache_scaled_delta(base_layer.getExecutionTarget()),
      cache_casted_delta(base_layer.getExecutionTarget()),
      cache_output(base_layer.getExecutionTarget()),
      cache_dy_scaled(base_layer.getExecutionTarget()),
      cache_dx_lora(base_layer.getExecutionTarget()),
      cache_dx_total(base_layer.getExecutionTarget())
{
    setMixedPrecision(base_layer.isMixedPrecisionEnabled());
    setAccumulated(base_layer.isAccumulated());
    setTrainable(base_layer.isTrainable());
    if (_rank > 0)
    {
        enableLoRA(_rank, _alpha, _dropout_rate, true);
    }
}

LoRA_Linear_Layer::LoRA_Linear_Layer(const LoRA_Linear_Layer &other)
    : ILayer(other),
      base_layer(other.base_layer),
      lora_a(other.lora_a),
      lora_b(other.lora_b),
      lora_dropout(other.lora_dropout ? std::make_unique<Dropout_Layer>(*other.lora_dropout) : nullptr),
      rank(other.rank),
      alpha(other.alpha),
      dropout_rate(other.dropout_rate),
      lora_enabled(other.lora_enabled),
      is_merged(other.is_merged),
      is_training_mode(other.is_training_mode),
      cache_has_dropout(other.cache_has_dropout),
      cache_scaled_delta(other.getExecutionTarget()),
      cache_casted_delta(other.getExecutionTarget()),
      cache_output(other.getExecutionTarget()),
      cache_dy_scaled(other.getExecutionTarget()),
      cache_dx_lora(other.getExecutionTarget()),
      cache_dx_total(other.getExecutionTarget())
{
}

LoRA_Linear_Layer &LoRA_Linear_Layer::operator=(const LoRA_Linear_Layer &other)
{
    if (this != &other)
    {
        ILayer::operator=(other);
        base_layer = other.base_layer;
        lora_a = other.lora_a;
        lora_b = other.lora_b;
        lora_dropout = other.lora_dropout ? std::make_unique<Dropout_Layer>(*other.lora_dropout) : nullptr;
        rank = other.rank;
        alpha = other.alpha;
        dropout_rate = other.dropout_rate;
        lora_enabled = other.lora_enabled;
        is_merged = other.is_merged;
        is_training_mode = other.is_training_mode;
        cache_has_dropout = other.cache_has_dropout;
        const Execution_Target target = other.getExecutionTarget();
        cache_scaled_delta = Tensor(target);
        cache_casted_delta = Tensor(target);
        cache_output = Tensor(target);
        cache_dy_scaled = Tensor(target);
        cache_dx_lora = Tensor(target);
        cache_dx_total = Tensor(target);
    }
    return *this;
}

void LoRA_Linear_Layer::initLoRAParams(size_t new_rank, float new_alpha, float new_dropout)
{
    if (new_rank == 0)
    {
        throw std::invalid_argument("LoRA_Linear_Layer::initLoRAParams: rank must be > 0");
    }
    if (!std::isfinite(new_alpha) || new_alpha <= 0.0f)
    {
        throw std::invalid_argument("LoRA_Linear_Layer::initLoRAParams: alpha must be finite and > 0");
    }
    if (!std::isfinite(new_dropout) || new_dropout < 0.0f || new_dropout >= 1.0f)
    {
        throw std::invalid_argument("LoRA_Linear_Layer::initLoRAParams: dropout_rate must be in [0, 1)");
    }
    const Execution_Target target = base_layer.getExecutionTarget();
    const Data_Type dtype = base_layer.isMixedPrecisionEnabled() ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    const size_t in_dim = base_layer.getInputDimension();
    const size_t out_dim = base_layer.getOutputDimension();

    lora_a = Linear_Layer(in_dim, new_rank, target, 1.0f, dtype, false);
    lora_b = Linear_Layer(new_rank, out_dim, target, 1.0f, dtype, false);
    Tensor b_zeros(Shape{ new_rank, out_dim }, std::vector<float>(new_rank * out_dim, 0.0f), target);
    lora_b.setWeights(b_zeros);

    lora_a.setMixedPrecision(base_layer.isMixedPrecisionEnabled());
    lora_b.setMixedPrecision(base_layer.isMixedPrecisionEnabled());
    lora_a.setAccumulated(is_accumulated);
    lora_b.setAccumulated(is_accumulated);
    lora_a.setTrainable(is_trainable);
    lora_b.setTrainable(is_trainable);

    if (new_dropout > 0.0f)
    {
        lora_dropout = std::make_unique<Dropout_Layer>(new_dropout, target);
        lora_dropout->setTrainingMode(is_training_mode);
    }
    else
    {
        lora_dropout.reset();
    }

    rank = new_rank;
    alpha = new_alpha;
    dropout_rate = new_dropout;
    lora_enabled = true;
    is_merged = false;
}

void LoRA_Linear_Layer::enableLoRA(size_t _rank, float _alpha, float _dropout_rate, bool _freeze_base)
{
    if (is_merged)
    {
        unmergeWeights();
    }
    if (base_layer.isTernary())
    {
        throw std::logic_error("LoRA_Linear_Layer::enableLoRA: Cannot enable LoRA on TERNARY_INT2 base layer");
    }
    initLoRAParams(_rank, _alpha, _dropout_rate);
    base_layer.setTrainable(!_freeze_base && is_trainable);
}

void LoRA_Linear_Layer::disableLoRA()
{
    if (is_merged)
    {
        unmergeWeights();
    }
    lora_enabled = false;
    rank = 0;
    lora_a = Linear_Layer();
    lora_b = Linear_Layer();
    lora_dropout.reset();
    base_layer.setTrainable(is_trainable);
}

void LoRA_Linear_Layer::mergeWeights()
{
    if (!isLoRAEnabled() || is_merged)
    {
        return;
    }
    Tensor ab(getExecutionTarget());
    lora_a.getWeights().matmul(lora_b.getWeights(), ab);
    Tensor scaled_ab(getExecutionTarget());
    ab.mulScalar(getScaling(), scaled_ab);
    Tensor cur_w = base_layer.getWeights();
    if (scaled_ab.getDataType() != cur_w.getDataType())
    {
        scaled_ab = scaled_ab.to(cur_w.getDataType());
    }
    Tensor merged_w(getExecutionTarget());
    cur_w.add(scaled_ab, merged_w);
    base_layer.setWeights(merged_w);
    is_merged = true;
}

void LoRA_Linear_Layer::unmergeWeights()
{
    if (!isLoRAEnabled() || !is_merged)
    {
        return;
    }
    Tensor ab(getExecutionTarget());
    lora_a.getWeights().matmul(lora_b.getWeights(), ab);
    Tensor scaled_ab(getExecutionTarget());
    ab.mulScalar(getScaling(), scaled_ab);
    Tensor cur_w = base_layer.getWeights();
    if (scaled_ab.getDataType() != cur_w.getDataType())
    {
        scaled_ab = scaled_ab.to(cur_w.getDataType());
    }
    Tensor restored_w(getExecutionTarget());
    cur_w.sub(scaled_ab, restored_w);
    base_layer.setWeights(restored_w);
    is_merged = false;
}

Tensor LoRA_Linear_Layer::forward(const Tensor &_input_tensor)
{
    if (!isLoRAEnabled() || is_merged)
    {
        return base_layer.forward(_input_tensor);
    }

    Tensor base_out = base_layer.forward(_input_tensor);
    cache_has_dropout = (lora_dropout != nullptr && is_training_mode && dropout_rate > 0.0f);
    Tensor mid = cache_has_dropout
        ? lora_a.forward(lora_dropout->forward(_input_tensor))
        : lora_a.forward(_input_tensor);
    Tensor delta = lora_b.forward(mid);
    delta.mulScalar(getScaling(), cache_scaled_delta);

    const Tensor *delta_ptr = &cache_scaled_delta;
    if (cache_scaled_delta.getDataType() != base_out.getDataType())
    {
        cache_scaled_delta.to(base_out.getDataType(), cache_casted_delta);
        delta_ptr = &cache_casted_delta;
    }
    base_out.add(*delta_ptr, cache_output);
    return cache_output;
}

Tensor LoRA_Linear_Layer::backward(const Tensor &_output_gradient)
{
    if (!isLoRAEnabled() || is_merged)
    {
        return base_layer.backward(_output_gradient);
    }

    Tensor dx_base = base_layer.backward(_output_gradient);
    _output_gradient.mulScalar(getScaling(), cache_dy_scaled);
    Tensor d_mid = lora_b.backward(cache_dy_scaled);
    Tensor dx_lora = lora_a.backward(d_mid);
    if (cache_has_dropout && lora_dropout != nullptr)
    {
        dx_lora = lora_dropout->backward(dx_lora);
    }

    const Tensor *lora_grad_ptr = &dx_lora;
    if (dx_lora.getDataType() != dx_base.getDataType())
    {
        dx_lora.to(dx_base.getDataType(), cache_dx_lora);
        lora_grad_ptr = &cache_dx_lora;
    }
    dx_base.add(*lora_grad_ptr, cache_dx_total);
    return cache_dx_total;
}

void LoRA_Linear_Layer::setTrainable(bool _trainable) noexcept
{
    ILayer::setTrainable(_trainable);
    if (!isLoRAEnabled())
    {
        base_layer.setTrainable(_trainable);
    }
    else
    {
        lora_a.setTrainable(_trainable);
        lora_b.setTrainable(_trainable);
    }
}

void LoRA_Linear_Layer::resetGradient()
{
    base_layer.resetGradient();
    if (isLoRAEnabled())
    {
        lora_a.resetGradient();
        lora_b.resetGradient();
        if (lora_dropout)
        {
            lora_dropout->resetGradient();
        }
    }
}

void LoRA_Linear_Layer::resetGradients()
{
    base_layer.resetGradients();
    if (isLoRAEnabled())
    {
        lora_a.resetGradients();
        lora_b.resetGradients();
        if (lora_dropout)
        {
            lora_dropout->resetGradient();
        }
    }
}

std::vector<std::pair<Tensor *, Tensor *>> LoRA_Linear_Layer::getParametersAndGradients()
{
    std::vector<std::pair<Tensor *, Tensor *>> params = base_layer.getParametersAndGradients();
    if (isLoRAEnabled() && !is_merged && is_trainable)
    {
        auto a_params = lora_a.getParametersAndGradients();
        auto b_params = lora_b.getParametersAndGradients();
        params.insert(params.end(), a_params.begin(), a_params.end());
        params.insert(params.end(), b_params.begin(), b_params.end());
    }
    return params;
}

std::unique_ptr<ILayer> LoRA_Linear_Layer::clone() const
{
    return std::make_unique<LoRA_Linear_Layer>(*this);
}

void LoRA_Linear_Layer::setExecutionTarget(Execution_Target _execution_target)
{
    if (getExecutionTarget() == _execution_target)
    {
        return;
    }
    logChangeExecutionTarget(_execution_target);
    base_layer.setExecutionTarget(_execution_target);
    if (isLoRAEnabled())
    {
        lora_a.setExecutionTarget(_execution_target);
        lora_b.setExecutionTarget(_execution_target);
        if (lora_dropout)
        {
            lora_dropout->setExecutionTarget(_execution_target);
        }
    }
    cache_scaled_delta = Tensor(_execution_target);
    cache_casted_delta = Tensor(_execution_target);
    cache_output = Tensor(_execution_target);
    cache_dy_scaled = Tensor(_execution_target);
    cache_dx_lora = Tensor(_execution_target);
    cache_dx_total = Tensor(_execution_target);
}

void LoRA_Linear_Layer::setAccumulated(bool _is_accumulated) noexcept
{
    ILayer::setAccumulated(_is_accumulated);
    base_layer.setAccumulated(_is_accumulated);
    if (isLoRAEnabled())
    {
        lora_a.setAccumulated(_is_accumulated);
        lora_b.setAccumulated(_is_accumulated);
    }
}

void LoRA_Linear_Layer::setTrainingMode(bool _is_training)
{
    is_training_mode = _is_training;
    base_layer.setTrainingMode(_is_training);
    if (lora_dropout)
    {
        lora_dropout->setTrainingMode(_is_training);
    }
}

void LoRA_Linear_Layer::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);
    base_layer.setMixedPrecision(_enable);
    if (isLoRAEnabled())
    {
        lora_a.setMixedPrecision(_enable);
        lora_b.setMixedPrecision(_enable);
    }
}

void LoRA_Linear_Layer::invalidateWeightCache() noexcept
{
    base_layer.invalidateWeightCache();
    if (isLoRAEnabled())
    {
        lora_a.invalidateWeightCache();
        lora_b.invalidateWeightCache();
    }
}

void LoRA_Linear_Layer::quantizeToTernary()
{
    if (isLoRAEnabled())
    {
        throw std::logic_error("LoRA_Linear_Layer::quantizeToTernary: Disable LoRA before quantizing base layer");
    }
    base_layer.quantizeToTernary();
}

void LoRA_Linear_Layer::dequantizeFromTernary()
{
    base_layer.dequantizeFromTernary();
}

void LoRA_Linear_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("LoRA_Linear_Layer::saveConfiguration: Output stream is not open");
    }
    base_layer.saveConfiguration(_output_file_stream);
    const uint32_t saved_rank = static_cast<uint32_t>(rank);
    uint8_t flags = 0;
    if (base_layer.hasBias()) flags |= 0x01u;
    if (lora_enabled) flags |= 0x02u;
    if (is_merged) flags |= 0x04u;
    if (base_layer.isTrainable()) flags |= 0x08u;
    if (is_trainable) flags |= 0x10u;
    _output_file_stream.write(reinterpret_cast<const char *>(&saved_rank), sizeof(saved_rank));
    _output_file_stream.write(reinterpret_cast<const char *>(&alpha), sizeof(alpha));
    _output_file_stream.write(reinterpret_cast<const char *>(&dropout_rate), sizeof(dropout_rate));
    _output_file_stream.write(reinterpret_cast<const char *>(&flags), sizeof(flags));
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("LoRA_Linear_Layer::saveConfiguration: Failed writing configuration");
    }
}

void LoRA_Linear_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    base_layer.saveInference(_output_file_stream);
    const uint8_t has_lora = isLoRAEnabled() ? 1u : 0u;
    _output_file_stream.write(reinterpret_cast<const char *>(&has_lora), sizeof(has_lora));
    if (has_lora != 0u)
    {
        lora_a.saveInference(_output_file_stream);
        lora_b.saveInference(_output_file_stream);
    }
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("LoRA_Linear_Layer::saveInference: Failed writing state");
    }
}

void LoRA_Linear_Layer::loadInference(std::ifstream &_input_file_stream)
{
    base_layer.loadInference(_input_file_stream);
    uint8_t has_lora = 0u;
    if (!_input_file_stream.read(reinterpret_cast<char *>(&has_lora), sizeof(has_lora)))
    {
        throw std::runtime_error("LoRA_Linear_Layer::loadInference: Failed reading LoRA flag");
    }
    if (has_lora != 0u)
    {
        lora_a.loadInference(_input_file_stream);
        lora_b.loadInference(_input_file_stream);
        rank = lora_a.getOutputDimension();
        lora_enabled = (rank > 0);
    }
}

void LoRA_Linear_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    base_layer.saveCheckpoint(_output_file_stream);
    const uint8_t has_lora = isLoRAEnabled() ? 1u : 0u;
    _output_file_stream.write(reinterpret_cast<const char *>(&has_lora), sizeof(has_lora));
    if (has_lora != 0u)
    {
        lora_a.saveCheckpoint(_output_file_stream);
        lora_b.saveCheckpoint(_output_file_stream);
    }
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("LoRA_Linear_Layer::saveCheckpoint: Failed writing checkpoint state");
    }
}

void LoRA_Linear_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    base_layer.loadCheckpoint(_input_file_stream);
    uint8_t has_lora = 0u;
    if (!_input_file_stream.read(reinterpret_cast<char *>(&has_lora), sizeof(has_lora)))
    {
        throw std::runtime_error("LoRA_Linear_Layer::loadCheckpoint: Failed reading LoRA flag");
    }
    if (has_lora != 0u)
    {
        lora_a.loadCheckpoint(_input_file_stream);
        lora_b.loadCheckpoint(_input_file_stream);
        rank = lora_a.getOutputDimension();
        lora_enabled = (rank > 0);
    }
}
