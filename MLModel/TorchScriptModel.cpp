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
  c10::optional<torch::Tensor> forces_tensor = torch::nullopt;

  auto const output = module_.forward(model_inputs_);

  if (output.isTuple())
  {
    auto const output_tuple = output.toTuple()->elements();
    if (output_tuple.empty())
    {
      throw std::runtime_error("TorchScript model returned an empty tuple.");
    }

    partial_energy_tensor = output_tuple[0].toTensor();

    if (output_tuple.size() > 1)
    {
      forces_tensor = output_tuple[1].toTensor();
    }
  }
  else
  {
    partial_energy_tensor = output.toTensor();
  }

  // Expand particle energies to full particle order.
  // The final input uses 0 for contributing particles.
  if (partial_energy_tensor.dim() != 0 && grad_idx >= 0
      && model_inputs_.size() > 1)
  {
    auto const particle_count
        = model_inputs_[grad_idx].toTensor().size(0);
    auto const values = partial_energy_tensor.reshape({-1});
    auto const mask
        = model_inputs_.back().toTensor().reshape({-1}).eq(0);

    if (mask.numel() != particle_count)
    {
      throw std::runtime_error(
          "Contribution mask length does not match particle count.");
    }

    if (values.numel() == particle_count)
    {
      // Full output: zero non-contributing particle energies.
      partial_energy_tensor
          = values * mask.to(values.scalar_type());
    }
    else
    {
      auto const contributing_count
          = mask.sum().item<std::int64_t>();

      if (values.numel() != contributing_count)
      {
        throw std::runtime_error(
            "Unexpected number of particle energies returned by model.");
      }

      // Packed output: scatter into the full particle ordering.
      partial_energy_tensor
          = torch::zeros({particle_count}, values.options())
                .masked_scatter(mask, values);
    }
  }

  auto const energy_tensor = partial_energy_tensor.sum();

  if (backprop)
  {
    if (grad_idx < 0
        || grad_idx >= static_cast<int>(model_inputs_.size()))
    {
      throw std::runtime_error(
          "Backpropagation requested but no input was marked with "
          "requires_grad (grad_idx = " + std::to_string(grad_idx) + ").");
    }

    energy_tensor.backward();
    forces_tensor = -model_inputs_[grad_idx].toTensor().grad();
  }

  if (energy)
  {
    *energy = energy_tensor.to(torch::kCPU).item<double>();
  }

  if (partial_energy)
  {
    if (partial_energy_tensor.dim() == 0)
    {
      throw std::runtime_error(
          "Requested partial energy, but model only provided a scalar.");
    }

    auto const cpu_particle_energy
        = partial_energy_tensor.to(torch::kFloat64)
              .to(torch::kCPU)
              .contiguous();

    std::memcpy(partial_energy,
                cpu_particle_energy.data_ptr<double>(),
                cpu_particle_energy.numel() * sizeof(double));
  }

  if (forces)
  {
    if (!forces_tensor.has_value())
    {
      throw std::runtime_error(
          "Forces requested, but neither the model provided forces nor "
          "backpropagation was requested.");
    }

    auto const cpu_forces
        = forces_tensor->to(torch::kFloat64)
              .to(torch::kCPU)
              .contiguous();

    std::memcpy(forces,
                cpu_forces.data_ptr<double>(),
                cpu_forces.numel() * sizeof(double));
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

void TorchScriptModel::SetAndScaleInputNode(
    int model_input_index,
    double * input,
    std::vector<std::int64_t> & size,
    bool requires_grad,
    bool clone,
    double scale_factor)
{
  auto scale_tensor = torch::tensor(
      scale_factor, torch::TensorOptions().dtype(model_precision_));
  scale_tensor = scale_tensor.to(*device_);
  SetInputNodeTemplate(model_input_index, input, size, requires_grad, clone);
  model_inputs_[model_input_index]
      = model_inputs_[model_input_index].toTensor() * scale_tensor;
  if (requires_grad)
  {
    model_inputs_[model_input_index].toTensor().retain_grad();
  }
}

void TorchScriptModel::WriteMLModel(std::string & model_path)
{
  module_.save(model_path);
}
