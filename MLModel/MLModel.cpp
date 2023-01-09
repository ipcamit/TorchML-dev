#include <map>
#include <string>
#include <iostream>
#include <stdexcept>
#include "MLModel.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <torch/script.h>

MLModel *MLModel::create(const char *model_file_path, MLModelType ml_model_type,
                         const char *const device_name, const int model_input_size) {
    if (ml_model_type == ML_MODEL_PYTORCH) {
        return new PytorchModel(model_file_path, device_name, model_input_size);
    } else {
        throw std::invalid_argument("Model Type Not defined");
    }
}

void PytorchModel::SetExecutionDevice(const char *const device_name) {
    // Use the requested device name char array to create a torch Device
    // object.  Generally, the ``device_name`` parameter is going to come
    // from a call to std::getenv(), so it is defined as const.

    std::string device_name_as_str;

    // Default to 'cpu'
    if (device_name == nullptr) {
        device_name_as_str = "cpu";
    } else {
        device_name_as_str = device_name;

        //Only compile if MPI is detected
        //n devices for n ranks, it will crash if MPI != GPU
        // TODO: Add a check if GPU aware MPI can be used
        #ifdef USE_MPI
        int rank=0, size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        auto kim_model_mpi_aware_env_var = std::getenv("KIM_MODEL_MPI_AWARE");
        if ((kim_model_mpi_aware_env_var != NULL) && (strcmp(kim_model_mpi_aware_env_var, "yes") == 0)){
            device_name_as_str += ":";
            device_name_as_str += std::to_string(rank);
        }
        #endif
    }
    device_ = new torch::Device(device_name_as_str);
}



torch::Dtype PytorchModel::get_torch_data_type(int *) {
    // Get the size used by 'int' on this platform and set torch tensor type
    // appropriately
    const std::size_t platform_size_int = sizeof(int);

    std::map<int, torch::Dtype> platform_size_int_to_torch_dtype;

    platform_size_int_to_torch_dtype[1] = torch::kInt8;
    platform_size_int_to_torch_dtype[2] = torch::kInt16;
    platform_size_int_to_torch_dtype[4] = torch::kInt32;
    platform_size_int_to_torch_dtype[8] = torch::kInt64;

    torch::Dtype torch_dtype =
            platform_size_int_to_torch_dtype[platform_size_int];

    return torch_dtype;
}

torch::Dtype PytorchModel::get_torch_data_type(double *) {
    return torch::kFloat64;
}

// TODO: Find a way to template SetInputNode...there are multiple definitions
// below that are exactly the same.  Since even derived implementations are
// virtual functions are also virtual, we can't use regular C++ templates.  Is
// it worth using the preprocessor for this?
void PytorchModel::SetInputNode(int model_input_index, int *input, int size,
                                bool requires_grad) {
    // Map C++ data type used for the input here into the appropriate
    // fixed-width torch data type
    torch::Dtype torch_dtype = get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::TensorOptions tensor_options =
            torch::TensorOptions().dtype(torch_dtype);

    // Finally, create the input tensor and store it on the relevant MLModel attr
    torch::Tensor input_tensor =
            torch::from_blob(input, {size}, tensor_options).to(*device_);

    // Workaround for non-leaf tensor thing, originating from .to(device) movement
    // https://discuss.pytorch.org/t/allocation-of-tensor-on-cuda-fails/144204/2
    if (requires_grad){
        input_tensor.retain_grad();
    }

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::SetInputNode(int model_input_index, int64_t *input, int size,
                                bool requires_grad) {
    // Map C++ data type used for the input here into the appropriate
    // fixed-width torch data type
    torch::Dtype torch_dtype = torch::kInt64;//get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::TensorOptions tensor_options =
            torch::TensorOptions().dtype(torch_dtype);

    // Finally, create the input tensor and store it on the relevant MLModel
    // attr
    torch::Tensor input_tensor =
            torch::from_blob(input, {size}, tensor_options).to(*device_);

    // Workaround for non-leaf tensor thing, originating from .to(device) movement
    // https://discuss.pytorch.org/t/allocation-of-tensor-on-cuda-fails/144204/2
    if (requires_grad){
        input_tensor.retain_grad();
    }

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::SetInputNode(int model_input_index, double *input, int size,
                                bool requires_grad) {
    // Map C++ data type used for the input here into the appropriate
    // fixed-width torch data type
    torch::Dtype torch_dtype = get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::TensorOptions tensor_options =
            torch::TensorOptions().dtype(torch_dtype).requires_grad(requires_grad);

    // Finally, create the input tensor and store it on the relevant MLModel
    // attr
    torch::Tensor input_tensor =
            torch::from_blob(input, {size}, tensor_options).to(*device_);

    // Workaround for non-leaf tensor thing, originating from .to(device) movement
    // https://discuss.pytorch.org/t/allocation-of-tensor-on-cuda-fails/144204/2
    if (requires_grad){
        input_tensor.retain_grad();
    }

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::SetInputNode(int model_input_index, double *input, std::vector<int> &size,
                                bool requires_grad) {

    torch::Dtype torch_dtype = get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::TensorOptions tensor_options =
            torch::TensorOptions().dtype(torch_dtype).requires_grad(requires_grad);

    // Finally, create the input tensor and store it on the relevant MLModel attr
    std::vector<int64_t> size_t;
    for (auto val: size) size_t.push_back(static_cast<int64_t>(val));
    torch::Tensor input_tensor =
            torch::from_blob(input, size_t, tensor_options).to(*device_);

    // Workaround for non-leaf tensor thing, originating from .to(device) movement
    // https://discuss.pytorch.org/t/allocation-of-tensor-on-cuda-fails/144204/2
    if (requires_grad){
        input_tensor.retain_grad();
    }

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::SetInputNode(int model_input_index, int layer,
                                int size, long **const unrolled_graph) {
    long *edge_index_array = unrolled_graph[layer];
    torch::Dtype torch_dtype = torch::kLong;//get_torch_data_type(edge_index_array);
    torch::TensorOptions tensor_options = torch::TensorOptions().dtype(torch_dtype);
    torch::Tensor edge_index =
            torch::from_blob(edge_index_array, {2, size}, tensor_options).to(*device_);
    model_inputs_[model_input_index] = edge_index;
}


void PytorchModel::Run(c10::IValue &out_tensor) {
    // FIXME: Make this work for arbitrary number/type of outputs?  This may
    // lead us to make Run() take no parameters, and instead define separate
    // methods for accessing each of the outputs of the ML model.

    // Run ML model's `forward` method and retrieve outputs as tuple
    // IMPORTANT: We require that the pytorch model's `forward`
    // method return a tuple where the energy is the first entry and
    // the forces are the second

    out_tensor = module_.forward(model_inputs_);
}

PytorchModel::PytorchModel(const char *model_file_path, const char *device_name, const int size_) {
    model_file_path_ = model_file_path;
    SetExecutionDevice(device_name);
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(model_file_path_,*device_);
    }
    catch (const c10::Error &e) {
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
}

void PytorchModel::GetInputNode(c10::IValue &out_tensor) {
    // return first tensor with grad = True
    for (auto &Ival: model_inputs_) {
        if (Ival.toTensor().requires_grad()) {
            out_tensor = Ival;
            return;
        }
    }
}

void PytorchModel::GetInputNode(int index, c10::IValue &out_tensor) {
    // return first tensor with grad = True
    out_tensor = model_inputs_[index];
}

void PytorchModel::SetInputSize(int size) {
    model_inputs_.resize(size);
}

PytorchModel::~PytorchModel() {
    delete device_;
}