#include "TorchScriptModel.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/csrc/jit/runtime/graph_executor.h>

void TorchScriptModel::Run(double * energy,
                           double * partial_energy,
                           double * forces,
                           bool backprop)
{
  torch::Tensor partial_energy_tensor;
  torch::Tensor energy_tensor;
  c10::optional<torch::Tensor> forces_tensor = torch::nullopt;

  auto out_tensor = module_.forward(model_inputs_);

  if (out_tensor.isTuple())
  {
    auto out_tensor_tuple = out_tensor.toTuple()->elements();
    partial_energy_tensor = out_tensor_tuple[0].toTensor();

    if (out_tensor_tuple.size() > 1)
    {
      forces_tensor = out_tensor_tuple[1].toTensor();
    }
  }
  else { partial_energy_tensor = out_tensor.toTensor(); }

  energy_tensor = partial_energy_tensor.sum();

  if (backprop)
  {
    if (grad_idx < 0 || grad_idx >= static_cast<int>(model_inputs_.size()))
    {
      throw std::runtime_error(
          "Backpropagation requested but no input was marked with "
          "requires_grad (grad_idx = " + std::to_string(grad_idx) + ").");
    }
    energy_tensor.backward();
    forces_tensor = -model_inputs_[grad_idx].toTensor().grad();
  }

  if (energy) { *energy = energy_tensor.to(torch::kCPU).item<double>(); }

  if (partial_energy)
  {
    if (partial_energy_tensor.dim() == 0)
    {
      throw std::runtime_error(
          "Requested partial energy, but model only provided a scalar.");
    }

    std::memcpy(
        partial_energy,
        partial_energy_tensor.to(torch::kFloat64)
            .to(torch::kCPU)
            .contiguous()
            .data_ptr<double>(),
        partial_energy_tensor.numel() * sizeof(double));
  }

  if (forces)
  {
    if (!forces_tensor.has_value())
    {
      throw std::runtime_error("Forces requested, but neither model provides "
                               "forces nor backpropagation was requested.");
    }

    std::memcpy(forces,
                forces_tensor->to(torch::kFloat64)
                    .to(torch::kCPU)
                    .contiguous()
                    .data_ptr<double>(),
                forces_tensor->numel() * sizeof(double));
  }
}

TorchScriptModel::TorchScriptModel(std::string & model_file_path,
                                   std::string & device_name,
                                   const int size_)
{
  model_file_path_ = model_file_path;
  device_ = std::make_unique<torch::Device>(
      torch_utils::SetExecutionDevice(device_name));
  model_precision_ = torch_utils::SetModelPrecisionFromEnv();
  try
  {
    module_ = torch::jit::load(model_file_path_, *device_);
  }
  catch (const c10::Error &)
  {
    std::cerr << "ERROR: An error occurred while attempting to load the "
                 "pytorch model file from path "
              << model_file_path << std::endl;
    throw;
  }

  module_.to(*device_);
  module_.to(model_precision_);

  model_inputs_.resize(size_);

  module_.eval();
  module_ = torch::jit::freeze(module_);

  torch::jit::FusionStrategy strategy;
  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  torch::jit::setFusionStrategy(strategy);
}

void TorchScriptModel::SetInputNode(int idx,
                                    int * const data,
                                    std::vector<std::int64_t> & size,
                                    bool requires_grad,
                                    bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}

void TorchScriptModel::SetInputNode(int idx,
                                    std::int64_t * const data,
                                    std::vector<std::int64_t> & size,
                                    bool requires_grad,
                                    bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}

void TorchScriptModel::SetInputNode(int idx,
                                    double * const data,
                                    std::vector<std::int64_t> & size,
                                    bool requires_grad,
                                    bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}

void TorchScriptModel::SetAndScaleInputNode(int model_input_index,
                                            double * input,
                                            std::vector<std::int64_t> & size,
                                            bool requires_grad,
                                            bool clone,
                                            double scale_factor)
{
   torch::Tensor scale_tensor = torch::tensor(scale_factor, torch::TensorOptions().dtype(model_precision_));
   scale_tensor = scale_tensor.to(*device_);
   SetInputNodeTemplate(model_input_index, input, size, requires_grad, clone);
   model_inputs_[model_input_index] = model_inputs_[model_input_index].toTensor() * scale_tensor;
  // on device multiplication, no need to copy back to CPU, should be faster
  // any better fix?
}

void TorchScriptModel::WriteMLModel(std::string & model_path)
{
  module_.save(model_path);
}
