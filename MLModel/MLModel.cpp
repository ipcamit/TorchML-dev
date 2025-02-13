#include "MLModel.hpp"
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#ifdef USE_MPI
#include <algorithm>
#include <mpi.h>
#include <unistd.h>
#endif

#include <torch/script.h>

MLModel * MLModel::create(std::string& model_file_path,
                          std::string& device_name,
                          const int model_input_size)
{
    return new PytorchModel(model_file_path, device_name, model_input_size);
}

void PytorchModel::SetExecutionDevice(std::string& device_name)
{
  // Use the requested device name char array to create a torch Device
  // object.  Generally, the ``device_name`` parameter is going to come
  // from a call to std::getenv(), so it is defined as const.

  std::string device_name_as_str;

  // Default to 'cpu'
  if (device_name.empty()) { device_name_as_str = "cpu"; }
  else
  {
    device_name_as_str = device_name;

// Only compile if MPI is detected
// n devices for n ranks, it will crash if MPI != GPU
//  TODO: Add a check if GPU aware MPI can be used
#ifdef USE_MPI
    std::cout << "INFO: Using MPI aware GPU allocation" << std::endl;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // get number of cuda devices visible
    auto cuda_device_visible_env_var
        = std::getenv("CUDA_VISIBLE_DEVICES");  // input "0,1,2"
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
    // assign cuda device to ranks in round-robin fashion
    device_name_as_str += ":";
    device_name_as_str
        += cuda_device_visible_ids[rank % num_cuda_devices_visible];
    char hostname[256];
    gethostname(hostname, 256);
    // poor man's sync print
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
  device_ = std::make_unique<torch::Device>(device_name_as_str);
}



void PytorchModel::Run(double * energy, double * partial_energy, double * forces, bool backprop)
{
  auto out_tensor = module_.forward(model_inputs_).toTuple()->elements();



}

PytorchModel::PytorchModel(std::string& model_file_path,
                           std::string& device_name,
                           const int size_)
{
  model_file_path_ = model_file_path;
  SetExecutionDevice(device_name);
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module_ = torch::jit::load(model_file_path_, *device_);
  }
  catch (const c10::Error & e)
  {
    std::cerr << "ERROR: An error occurred while attempting to load the "
                 "pytorch model file from path "
              << model_file_path << std::endl;
    throw;
  }

  SetExecutionDevice(device_name);

  // Copy model to execution device
  module_.to(*device_);

  // Reserve size for the four fixed model inputs (particle_contributing,
  // coordinates, number_of_neighbors, neighbor_list)
  // Model inputs to be determined
  model_inputs_.resize(size_);

  // Set model to evaluation mode to set any dropout or batch normalization
  // layers to evaluation mode
  module_.eval();
  module_ = torch::jit::freeze(module_);

  torch::jit::FusionStrategy strategy;
  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  torch::jit::setFusionStrategy(strategy);
}


void PytorchModel::SetInputSize(int size) { model_inputs_.resize(size); }

void PytorchModel::SetInputNode( int idx, int * const data, std::vector<std::int64_t> & size, bool requires_grad, bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}
void PytorchModel::SetInputNode( int idx, int64_t * const data, std::vector<std::int64_t> & size, bool requires_grad, bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}
void PytorchModel::SetInputNode( int idx, double * const data, std::vector<std::int64_t> & size, bool requires_grad, bool clone)
{
  SetInputNodeTemplate(idx, data, size, requires_grad, clone);
}
