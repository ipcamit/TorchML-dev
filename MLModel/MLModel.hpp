#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>

#ifndef DISABLE_GRAPH
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#endif


// common datatype to torch type map.
template<typename T>
torch::Dtype getTorchDtype()
{
  if (std::is_same<T, int>::value) return torch::kInt32;
  if (std::is_same<T, std::int64_t>::value) return torch::kInt64;
  if (std::is_same<T, float>::value) return torch::kFloat32;
  if (std::is_same<T, double>::value) return torch::kFloat64;
  throw std::runtime_error("Invalid datatype provided as input to the model");
}

/* Abstract base class for an ML model -- 'product' of the factory pattern */
class MLModel
{
 public:
  static MLModel * create(std::string & /*model_file_path*/,
                          std::string & /*device_name*/,
                          int /*model_input_size*/);

  virtual void SetInputNode(int /*model_input_index*/,
                            int * /*input*/,
                            std::vector<std::int64_t> & /*size*/,
                            bool /*requires grad*/,
                            bool /* to clone*/)
      = 0;

  virtual void SetInputNode(int /*model_input_index*/,
                            std::int64_t * /*input*/,
                            std::vector<std::int64_t> & /*size*/,
                            bool /*requires grad*/,
                            bool /* to clone*/)
      = 0;

  virtual void SetInputNode(int /*model_input_index*/,
                            double * /*input*/,
                            std::vector<std::int64_t> & /*size*/,
                            bool /*requires grad*/,
                            bool /* to clone*/)
      = 0;


  virtual void Run(double *, double *, double *, bool) = 0;

  virtual void WriteMLModel(std::string & /*model_path*/) = 0;

  virtual ~MLModel() = default;
};

// Concrete MLModel corresponding to pytorch
class PytorchModel : public MLModel
{
 private:
  torch::jit::script::Module module_;
  std::vector<torch::jit::IValue> model_inputs_;
  std::unique_ptr<torch::Device> device_;

  void SetExecutionDevice(std::string & /*device_name*/);
  int grad_idx;

  template<typename T>
  void SetInputNodeTemplate(int idx,
                            T * data,
                            std::vector<std::int64_t> & shape,
                            bool requires_grad,
                            bool clone)
  {
    // Configure tensor options
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(getTorchDtype<T>())
                                       .requires_grad(requires_grad);

    // Create tensor from blob
    torch::Tensor input_tensor = torch::from_blob(data, shape, options);

    // Only need clone if device is CPU, else implicit deep copy will be
    // triggered
    if (clone || (*device_ == torch::kCPU)) input_tensor = input_tensor.clone();

    // explicit copy to device if not done already
    if (input_tensor.device() != *device_)
      input_tensor = input_tensor.to(*device_);

    // Workaround for PyTorch bug
    if (requires_grad)
    {
      input_tensor.retain_grad();
      grad_idx = idx;
    }

    model_inputs_[idx] = input_tensor;
  }


 public:
  std::string model_file_path_;

  PytorchModel(std::string & /*model_file_path*/,
               std::string & /*device_name*/,
               int /*input size*/);

  void SetInputNode(int /*model_input_index*/,
                    int * /*input*/,
                    std::vector<std::int64_t> & /*size*/,
                    bool /*requires grad*/,
                    bool /*to clone or not*/) override;

  void SetInputNode(int /*model_input_index*/,
                    std::int64_t * /*input*/,
                    std::vector<std::int64_t> & /*size*/,
                    bool /*requires grad*/,
                    bool /*to clone or not*/) override;

  void SetInputNode(int /*model_input_index*/,
                    double * /*input*/,
                    std::vector<std::int64_t> & /*size*/,
                    bool /*requires grad*/,
                    bool /*to clone or not*/) override;

  void Run(double * /*energy*/,
           double * /*partial_energy*/,
           double * /*forces*/,
           bool /*backprop*/) override;

  void WriteMLModel(std::string & /*path*/) override;

  ~PytorchModel() override = default;
};

#endif /* MLMODEL_HPP */