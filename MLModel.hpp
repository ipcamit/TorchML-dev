#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <cstdlib>
#include <vector>
#include <string>

#include <torch/script.h>

// TODO Specify kind of models and enumerate
// Basic kind I can think of
// 1. Core model: numberOfParticles, coord, neighbours count, neighbour list
// 2. Descriptor model: numberOfParticles, coord, neighbours count, neighbour list
// 3. Graph model: TBA
enum MLModelType {
    ML_MODEL_PYTORCH,
};

/* Abstract base class for an ML model -- 'product' of the factory pattern */
class MLModel {
public:
    static MLModel *create(const char * /*model_file_path*/,
                           MLModelType /*ml_model_type*/,
                           const char * /*device_name*/,
                           int /*model_input_size*/);

    // TODO: Should we use named inputs instead?  I believe they're required
    // by ONNX, but not sure exactly how they work vis-a-vis exporting to a
    // torchscript file.

    // Function templates can't be used for pure virtual functions, and since
    // SetInputNode and Run each have their own (different) support argument
    // types, we can't use a class template.  So, we explicitly define each
    // supported overloading.
    virtual void SetInputNode(int /*model_input_index*/, int * /*input*/,
                              int /*size*/, bool requires_grad = false) = 0;

    virtual void SetInputNode(int /*model_input_index*/, double * /*input*/,
                              int /*size*/, bool requires_grad = false) = 0;

    virtual void SetInputNode(int /*model_input_index*/, double * /*input*/,
                      std::vector<int>& /*arb size*/, bool requires_grad = false) = 0;

    virtual void GetInputNode(int /*model_input_index*/, c10::IValue &) = 0;
    virtual void GetInputNode( c10::IValue &) = 0;

    virtual void Run(c10::IValue &) = 0;

    virtual ~MLModel() {};
};

// Concrete MLModel corresponding to pytorch
class PytorchModel : public MLModel {
private:
    torch::jit::script::Module module_;
    std::vector<torch::jit::IValue> model_inputs_;
    torch::Device *device_;

    torch::Dtype get_torch_data_type(int *);

    torch::Dtype get_torch_data_type(double *);

    void SetExecutionDevice(const char * /*device_name*/);

public:
    const char *model_file_path_;
    // TODO move descriptor to param file as ideally model is not dependent on it
//    bool descriptor_required = false;
//    std::string descriptor_function;

    PytorchModel(const char * /*model_file_path*/,
                 const char * /*device_name*/,
                 int /*input size*/);

    void SetInputNode(int /*model_input_index*/, int * /*input*/, int /*size*/,
                      bool requires_grad = false);

    void SetInputNode(int /*model_input_index*/, double * /*input*/,
                      int /*size*/, bool requires_grad = false);

    void SetInputNode(int /*model_input_index*/, double * /*input*/,
                      std::vector<int>& /*arb size*/, bool requires_grad = false);

    void GetInputNode(int /*model_input_index*/, c10::IValue &);
    void GetInputNode(c10::IValue &);

    void SetInputSize(int /*input size*/);

//    void Run(double * /*energy*/, double * /*forces*/);
    void Run(c10::IValue&);

    ~PytorchModel();
};

#endif /* MLMODEL_HPP */