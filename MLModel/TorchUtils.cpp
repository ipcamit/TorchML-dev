#include "TorchUtils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace torch_utils {

torch::Device SetExecutionDevice(std::string const & device_name)
{
  std::string device_name_as_str;

  if (device_name.empty()) { device_name_as_str = "cpu"; }
  else
  {
    device_name_as_str = device_name;

#ifdef USE_MPI
    std::cout << "INFO: Using MPI aware GPU allocation" << std::endl;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto cuda_device_visible_env_var = std::getenv("CUDA_VISIBLE_DEVICES");
    std::vector<std::string> cuda_device_visible_ids;
    int num_cuda_devices_visible = 0;
    if (cuda_device_visible_env_var != nullptr)
    {
      std::string cuda_device_visible_env_var_str(cuda_device_visible_env_var);
      num_cuda_devices_visible
          = std::count(cuda_device_visible_env_var_str.begin(),
                       cuda_device_visible_env_var_str.end(),
                       ',')
            + 1;
      for (int i = 0; i < num_cuda_devices_visible; i++)
      {
        cuda_device_visible_ids.push_back(
            cuda_device_visible_env_var_str.substr(
                0, cuda_device_visible_env_var_str.find(',')));
        cuda_device_visible_env_var_str.erase(
            0, cuda_device_visible_env_var_str.find(',') + 1);
      }
    }
    else
    {
      throw std::invalid_argument(
          "CUDA_VISIBLE_DEVICES not set\n "
          "You requested for manual MPI aware device allocation but "
          "CUDA_VISIBLE_DEVICES is not set\n");
    }
    device_name_as_str += ":";
    device_name_as_str
        += cuda_device_visible_ids[rank % num_cuda_devices_visible];
    char hostname[256];
    gethostname(hostname, 256);
    for (int i = 0; i < size; i++)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == rank)
      {
        std::cout << "INFO: Rank " << rank << " on " << hostname
                  << " is using device " << device_name_as_str << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
  }

  return torch::Device(device_name_as_str);
}

torch::Dtype SetModelPrecisionFromEnv()
{
  const char * precision_env = std::getenv("KIM_MODEL_PRECISION");
  if (precision_env == nullptr) { return torch::kFloat64; }

  std::string precision(precision_env);
  std::transform(precision.begin(),
                 precision.end(),
                 precision.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (precision == "double" || precision == "float64" || precision == "fp64")
  {
    return torch::kFloat64;
  }
  if (precision == "float" || precision == "float32" || precision == "single"
      || precision == "fp32")
  {
    return torch::kFloat32;
  }

  std::cerr << "WARNING: Unsupported KIM_MODEL_PRECISION='" << precision_env
            << "'. Falling back to 'double'." << std::endl;
  return torch::kFloat64;
}

}  // namespace torch_utils
