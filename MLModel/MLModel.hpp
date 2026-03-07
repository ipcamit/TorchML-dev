#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <cstdint>
#include <string>
#include <vector>

class MLModel
{
 public:
  virtual void SetInputNode(int model_input_index,
                            int * input,
                            std::vector<std::int64_t> & size,
                            bool requires_grad,
                            bool clone)
      = 0;

  virtual void SetInputNode(int model_input_index,
                            std::int64_t * input,
                            std::vector<std::int64_t> & size,
                            bool requires_grad,
                            bool clone)
      = 0;

  virtual void SetInputNode(int model_input_index,
                            double * input,
                            std::vector<std::int64_t> & size,
                            bool requires_grad,
                            bool clone)
      = 0;

  virtual void Run(double * energy,
                   double * partial_energy,
                   double * forces,
                   bool backprop)
      = 0;

  virtual void WriteMLModel(std::string & model_path) = 0;

  virtual ~MLModel() = default;
};

#endif /* MLMODEL_HPP */
