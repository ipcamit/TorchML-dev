#ifndef TORCHSCRIPTMODEL_HPP
#define TORCHSCRIPTMODEL_HPP

#include "MLModel.hpp"
#include "TorchUtils.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

class TorchScriptModel : public MLModel
{
 private:
  torch::jit::script::Module module_;
  std::vector<torch::jit::IValue> model_inputs_;
  std::unique_ptr<torch::Device> device_;
  torch::Dtype model_precision_ = torch::kFloat64;
  int grad_idx = -1;

  template<typename T>
  void SetInputNodeTemplate(int idx,
                            T * data,
                            std::vector<std::int64_t> & shape,
                            bool requires_grad,
    bool clone)
  {
    torch::TensorOptions options = torch::TensorOptions()
                                        .dtype(torch_utils::getTorchDtype<T>())
                                        .requires_grad(requires_grad);

    torch::Tensor input_tensor = torch::from_blob(data, shape, options);

    if constexpr (std::is_floating_point_v<T>)
    {
      if (input_tensor.dtype() != model_precision_)
      {
        input_tensor = input_tensor.to(model_precision_);
      }
    }

    if (input_tensor.device() != *device_)
      input_tensor = input_tensor.to(*device_);

    if (clone || (*device_ == torch::kCPU)) input_tensor = input_tensor.clone();

    if (requires_grad)
    {
      input_tensor.retain_grad();
      grad_idx = idx;
    }

    model_inputs_[idx] = input_tensor;
  }

 public:
  std::string model_file_path_;

  TorchScriptModel(std::string & model_file_path,
                   std::string & device_name,
                   int input_size);

  void SetInputNode(int model_input_index,
                    int * input,
                    std::vector<std::int64_t> & size,
                    bool requires_grad,
                    bool clone) override;

  void SetInputNode(int model_input_index,
                    std::int64_t * input,
                    std::vector<std::int64_t> & size,
                    bool requires_grad,
                    bool clone) override;

  void SetInputNode(int model_input_index,
                    double * input,
                    std::vector<std::int64_t> & size,
                    bool requires_grad,
                    bool clone) override;

  void SetAndScaleInputNode(int model_input_index,
                           double * input,
                           std::vector<std::int64_t> & size,
                           bool requires_grad,
                           bool clone,
                           double scale_factor) override;

  void Run(double * energy,
           double * partial_energy,
           double * forces,
           bool backprop) override;

  void WriteMLModel(std::string & path) override;

  ~TorchScriptModel() override = default;
};

#endif /* TORCHSCRIPTMODEL_HPP */
