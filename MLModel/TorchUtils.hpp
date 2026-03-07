#ifndef TORCHUTILS_HPP
#define TORCHUTILS_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <torch/torch.h>

namespace torch_utils {

torch::Device SetExecutionDevice(std::string const & device_name);

torch::Dtype SetModelPrecisionFromEnv();

template<typename T>
torch::Dtype getTorchDtype()
{
  if (std::is_same_v<T, int>) return torch::kInt32;
  if (std::is_same_v<T, std::int64_t>) return torch::kInt64;
  if (std::is_same_v<T, float>) return torch::kFloat32;
  if (std::is_same_v<T, double>) return torch::kFloat64;
  throw std::runtime_error("Invalid datatype provided as input to the model");
}

}  // namespace torch_utils

#endif /* TORCHUTILS_HPP */
