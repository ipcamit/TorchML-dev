#include "TorchExportModel.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

void TorchExportModel::Run(double * energy,
                           double * partial_energy,
                           double * forces,
                           bool backprop)
{
  auto outputs = loader_->run(model_inputs_);
  if (outputs.empty())
  {
    throw std::runtime_error("TorchExport model returned no outputs.");
  }

  auto partial_energy_tensor = outputs[0];

  // Expand particle energies to full particle order.
  // The final input uses 0 for contributing particles.
  if (partial_energy_tensor.dim() != 0 && grad_idx_ >= 0
      && model_inputs_.size() > 1)
  {
    auto const particle_count = model_inputs_[grad_idx_].size(0);
    auto const values = partial_energy_tensor.reshape({-1});
    auto const mask = model_inputs_.back().reshape({-1}).eq(0);

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
    if (outputs.size() <= 1)
    {
      throw std::runtime_error(
          backprop
              ? "Forces requested with backpropagation, but model did not "
                "provide gradient output."
              : "Forces requested, but model did not provide force output.");
    }

    auto const cpu_forces
        = outputs[1].to(torch::kFloat64)
              .to(torch::kCPU)
              .contiguous();

    std::memcpy(forces,
                cpu_forces.data_ptr<double>(),
                cpu_forces.numel() * sizeof(double));
  }
}


TorchExportModel::TorchExportModel(std::string & model_file_path,
                                   std::string & device_name,
                                   int size_)
{
  model_file_path_ = model_file_path;
  device_ = std::make_unique<torch::Device>(
      torch_utils::SetExecutionDevice(device_name));
  model_precision_ = torch_utils::SetModelPrecisionFromEnv();

  try // kernel for gpu *_gpu.pt2
  {
    loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        model_file_path_);
  }
  catch (const std::exception &)
  {
    std::cerr << "ERROR: An error occurred while attempting to load the "
                 "TorchExport model package from path "
              << model_file_path_ << std::endl;
    throw;
  }

  model_inputs_.resize(size_);
}

void TorchExportModel::SetInputNode(int idx,
                                    int * const data,
                                    std::vector<std::int64_t> & size,
                                    bool requires_grad,
                                    bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}

void TorchExportModel::SetInputNode(int idx,
                                    std::int64_t * const data,
                                    std::vector<std::int64_t> & size,
                                    bool requires_grad,
                                    bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}

void TorchExportModel::SetInputNode(int idx,
                                    double * const data,
                                    std::vector<std::int64_t> & size,
                                    bool requires_grad,
                                    bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}

void TorchExportModel::SetAndScaleInputNode(int idx,
                                    double * const data,
                                    std::vector<std::int64_t> & size,
                                    bool requires_grad,
                                    bool clone,
                                    double scale_factor)
{

  torch::Tensor scale_tensor = torch::tensor(scale_factor, torch::TensorOptions().dtype(model_precision_).device(*device_));
  scale_tensor = scale_tensor.to(*device_);
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
  model_inputs_[idx] = model_inputs_[idx] * scale_tensor;
}

void TorchExportModel::WriteMLModel(std::string & model_path)
{
  //TODO: try and copy older model?
  std::cerr << "The Model is of type '.pt2', CANNOT SAVE IT OR MUTATE IT\n";
  std::cerr << "model_path: " << model_path << std::endl;
}

