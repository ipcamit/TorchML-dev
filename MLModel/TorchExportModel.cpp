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

  torch::Tensor partial_energy_tensor = outputs[0];
  torch::Tensor energy_tensor = partial_energy_tensor.sum();

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
    if (outputs.size() > 1)
    {
      torch::Tensor forces_tensor = outputs[1];
      std::memcpy(forces,
                  forces_tensor.to(torch::kFloat64)
                      .to(torch::kCPU)
                      .contiguous()
                      .data_ptr<double>(),
                  forces_tensor.numel() * sizeof(double));
    }
    else if (backprop)
    {
      throw std::runtime_error(
          "Forces requested with backpropagation for TorchExport model, but "
          "this model did not provide gradient output.");
    }
    else
    {
      throw std::runtime_error(
          "Forces requested, but TorchExport model did not provide force "
          "output.");
    }
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

void TorchExportModel::WriteMLModel(std::string & model_path)
{
  //TODO: try and copy older model?
  std::cerr << "The Model is of type '.pt2', CANNOT SAVE IT OR MUTATE IT\n";
  std::cerr << "model_path: " << model_path << std::endl;
}
