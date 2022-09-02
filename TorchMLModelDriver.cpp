#include "TorchMLModelDriver.hpp"
//#include "TorchMLModelImplementation.hpp"
#include "KIM_LogMacros.hpp"
#include <map>
#include <vector>
#include <algorithm>

#define MAX_FILE_NUM 3
//typedef double VecOfSize3[3];
#define KIM_DEVICE_ENV_VAR "KIM_MODEL_EXECUTION_DEVICE"


//==============================================================================
//
// This is the standard interface to KIM Model Drivers
//
//==============================================================================

//******************************************************************************
extern "C" {
int model_driver_create(KIM::ModelDriverCreate *const modelDriverCreate,
                        KIM::LengthUnit const requestedLengthUnit,
                        KIM::EnergyUnit const requestedEnergyUnit,
                        KIM::ChargeUnit const requestedChargeUnit,
                        KIM::TemperatureUnit const requestedTemperatureUnit,
                        KIM::TimeUnit const requestedTimeUnit) {
    int ier;
    // read input files, convert units if needed, compute
    // interpolation coefficients, set cutoff, and publish parameters
    auto modelObject = new TorchMLModelDriver(modelDriverCreate,
                                              requestedLengthUnit,
                                              requestedEnergyUnit,
                                              requestedChargeUnit,
                                              requestedTemperatureUnit,
                                              requestedTimeUnit,
                                              &ier);

    if (ier) {
        // constructor already reported the error
        delete modelObject;
        return ier;
    }

    // register pointer to TorchMLModelDriver object in KIM object
    modelDriverCreate->SetModelBufferPointer(static_cast<void *>(modelObject));

    // everything is good
    ier = false;
    return ier;
}
}  // extern "C"

//==============================================================================
//
// Implementation of TorchMLModelDriver public wrapper functions
//
//==============================================================================

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

TorchMLModelDriver::TorchMLModelDriver(
        KIM::ModelDriverCreate *const modelDriverCreate,
        KIM::LengthUnit const requestedLengthUnit,
        KIM::EnergyUnit const requestedEnergyUnit,
        KIM::ChargeUnit const requestedChargeUnit,
        KIM::TemperatureUnit const requestedTemperatureUnit,
        KIM::TimeUnit const requestedTimeUnit,
        int *const ier) {
    std::cout << std::boolalpha;
    *ier = false;
    //initialize members to remove warning----
    influence_distance = 0.0;
    n_elements = 0;
    torchModel = nullptr;
    returns_forces = false;
    // Read parameter files from model driver ---------------------------------------
    // also initialize the torchModel
    readParameters(modelDriverCreate, ier);
    LOG_DEBUG("Read Param files");
    if (*ier) return;

    // Unit conversions -----------------------------------------------------------------
    unitConversion(modelDriverCreate, requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
                   requestedTemperatureUnit, requestedTimeUnit, ier);
    LOG_DEBUG("Registered Unit Conversion");
    if (*ier) return;

    // Set Influence distance ---------------------------------------------------------
    int modelWillNotRequestNeighborsOfNoncontributingParticles_;
    if (preprocessing == "Graph") {
        modelWillNotRequestNeighborsOfNoncontributingParticles_ = static_cast<int>(false);
    } else {
        modelWillNotRequestNeighborsOfNoncontributingParticles_ = static_cast<int>(true);
    }
    modelDriverCreate->SetInfluenceDistancePointer(&influence_distance);
    modelDriverCreate->SetNeighborListPointers(1, &cutoff_distance,
                                               &modelWillNotRequestNeighborsOfNoncontributingParticles_);

    // Species code --------------------------------------------------------------------
    setSpecies(modelDriverCreate, ier);
    LOG_DEBUG("Registered Species");
    if (*ier) return;

    // Register Index settings-----------------------------------------------------------
    modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);

    // Register Parameters --------------------------------------------------------------
    // All model parameters are inside model So nothing to do here? Just Registering
    // n_elements lest KIM complaints. TODO Discuss with Ryan?
    *ier = modelDriverCreate->SetParameterPointer(1, &n_elements, "n_elements", "slope");
    LOG_DEBUG("Registered Parameter");
    if (*ier) return;

    // Register function pointers -----------------------------------------------------------
    registerFunctionPointers(modelDriverCreate, ier);
    if (*ier) return;

    // Set Units----------------------------------------------------------------------------
    *ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                       requestedEnergyUnit,
                                       KIM::CHARGE_UNIT::unused,
                                       KIM::TEMPERATURE_UNIT::unused,
                                       KIM::TIME_UNIT::unused);

    // Set preprocessor descriptor callbacks --------------------------------------------------
    if (preprocessing == "Descriptor") {
        //TODO delete allocation
        descriptor = new Descriptor(descriptor_name, descriptor_param_file);
    } else if (preprocessing == "Graph") {
        graph_edge_indices = new int *[n_layers];
        for (int i = 0; i < n_layers; i++) graph_edge_indices[i] = nullptr;
    } else {
        descriptor = nullptr;
        graph_edge_indices = nullptr;
    }
    descriptor_array = nullptr;
}

//******************************************************************************
//TorchMLModelDriver::~TorchMLModelDriver() { delete implementation_; }

//******************************************************************************
// static member function
int TorchMLModelDriver::Destroy(KIM::ModelDestroy *const modelDestroy) {
    TorchMLModelDriver *modelObject;
    modelDestroy->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

    delete modelObject;

    // everything is good
    return false;
}

//******************************************************************************
// static member function
int TorchMLModelDriver::Refresh(KIM::ModelRefresh *const modelRefresh) {
    TorchMLModelDriver *modelObject;
    modelRefresh->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
    return false;
}

//******************************************************************************
// static member function
int TorchMLModelDriver::Compute(
        KIM::ModelCompute const *const modelCompute,
        KIM::ModelComputeArguments const *const modelComputeArguments) {

    TorchMLModelDriver *modelObject;
    modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));;
    modelObject->Run(modelComputeArguments);
    // TODO see proper way to return error codes
    return false;
}

//******************************************************************************
// static member function
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate

int TorchMLModelDriver::ComputeArgumentsCreate(
        KIM::ModelCompute const *const modelCompute,
        KIM::ModelComputeArgumentsCreate *const modelComputeArgumentsCreate) {
//    TorchMLModelDriver * modelObject;
//    modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
    LOG_INFORMATION("Compute argument create");
    int error = modelComputeArgumentsCreate->SetArgumentSupportStatus(
            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
            KIM::SUPPORT_STATUS::optional)
                || modelComputeArgumentsCreate->SetArgumentSupportStatus(
            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
            KIM::SUPPORT_STATUS::optional);
    // register callbacks
    LOG_INFORMATION("Register callback supportStatus");
//    error = error
//          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
//              KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
//              KIM::SUPPORT_STATUS::optional)
//          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
//              KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
//              KIM::SUPPORT_STATUS::optional);
    return error;
}

// *****************************************************************************
// Auxiliary methods------------------------------------------------------------

void TorchMLModelDriver::Run(const KIM::ModelComputeArguments *const modelComputeArguments) {
    c10::IValue out_tensor;
    preprocessInputs(modelComputeArguments);
    torchModel->Run(out_tensor);
    postprocessOutputs(out_tensor, modelComputeArguments);
}

// -----------------------------------------------------------------------------
void TorchMLModelDriver::preprocessInputs(KIM::ModelComputeArguments const *const modelComputeArguments) {
    //TODO: Make preprocessing type enums
    if (preprocessing == "None") {
        setDefaultInputs(modelComputeArguments);
    } else if (preprocessing == "Descriptor") {
        setDescriptorInputs(modelComputeArguments);
    } else if (preprocessing == "Graph") {
        setGraphInputs(modelComputeArguments);
    }
}

// -----------------------------------------------------------------------------
void TorchMLModelDriver::postprocessOutputs(c10::IValue &out_tensor,
                                            KIM::ModelComputeArguments const *modelComputeArguments) {

    double *energy = nullptr;
    double *forces = nullptr;
    int const *numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling
    int *particleContributing = nullptr;
    double *coordinates = nullptr;

    auto ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
            &numberOfParticlesPointer)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
            &particleSpeciesCodes)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
            &particleContributing)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::coordinates,
            (double const **) &coordinates)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
            (double const **) &forces)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
            &energy);

    if (ier) return;

    int contributing_atoms_count = 0;
    for (int i = 0; i < *numberOfParticlesPointer; i++) {
        if (*(particleContributing + i) == 1) {
            contributing_atoms_count += 1;
        }
    }

    if (returns_forces) {
        const auto output_tensor_list = out_tensor.toTuple()->elements();
        *energy = *output_tensor_list[0].toTensor().to(torch::kCPU).data_ptr<double>();
        auto torch_forces = output_tensor_list[1].toTensor().to(torch::kCPU);
        auto force_accessor = torch_forces.accessor<double, 1>();
        for (int atom_count = 0; atom_count < force_accessor.size(0); ++atom_count) {
            forces[atom_count] = -force_accessor[atom_count];
        }
    } else {
        std::cout << "SUMMING UP ENERGIES. THIS WILL BE REMOVED. SUM YOUR OWN ENERGIES\n";
        out_tensor.toTensor().sum().backward();
        c10::IValue input_tensor;
        torchModel->GetInputNode(input_tensor);
        auto input_grad = input_tensor.toTensor().grad();
        *energy = *out_tensor.toTensor().to(torch::kCPU).data_ptr<double>();
        int neigh_from = 0;
        int n_neigh = 0;
        if (preprocessing == "Descriptor") {
            // TODO this is temporary hard coded fix. Need to improve it by inheriting Base
            // descriptor class in all descriptors. **Priority**
            std::cout << "USING SYMUFUN WORKAROUND. FIX ME ASAP" << "\n";
            auto sf = reinterpret_cast<SymmetryFunctionParams *>(descriptor->descriptor_map["SymmetryFunction"]);
            int width = sf->width;
            //
            for (int i = 0; i < contributing_atoms_count; i++) {
                n_neigh = num_neighbors_[i];
                std::vector<int> n_list(neighbor_list.begin() + neigh_from,
                                        neighbor_list.begin() + neigh_from + n_neigh);
                neigh_from += n_neigh;
                grad_symmetry_function_atomic(i,
                                              coordinates,
                                              forces,
                                              particleSpeciesCodes,
                                              n_list.data(),
                                              n_neigh,
                                              input_tensor.toTensor().data_ptr<double>() + (i * width),
                                              input_grad.data_ptr<double>() + (i * width),
                                              sf);
            }
            for (int i = 0; i < *numberOfParticlesPointer; i++) {
                // forces = -grad
                *(forces + i) *= -1.0;
                *(forces + i + 1) *= -1.0;
                *(forces + i + 2) *= -1.0;
            }
        } else {
            auto force_accessor = input_grad.accessor<double, 1>();
            for (int atom_count = 0; atom_count < force_accessor.size(0); ++atom_count) {
                forces[atom_count] = -force_accessor[atom_count];
            }
        }
    }
}

// -----------------------------------------------------------------------------
void TorchMLModelDriver::updateNeighborList(KIM::ModelComputeArguments const *const modelComputeArguments,
                                            int const numberOfParticles) {
    int numOfNeighbors;
    int const *neighbors;
    num_neighbors_.clear();
    neighbor_list.clear();
    std::cout << numberOfParticles << "\n"; // HERE PRINT
    for (int i = 0; i < numberOfParticles; i++) {
        modelComputeArguments->GetNeighborList(0,
                                               i,
                                               &numOfNeighbors,
                                               &neighbors);
        num_neighbors_.push_back(numOfNeighbors);
        for (int neigh = 0; neigh < numOfNeighbors; neigh++) {
            neighbor_list.push_back(neighbors[neigh]);
        }
    }
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriver::setDefaultInputs(const KIM::ModelComputeArguments *modelComputeArguments) {
    int const *numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling
    int *particleContributing = nullptr;
    double *coordinates = nullptr;

    auto ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
            &numberOfParticlesPointer)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
            &particleSpeciesCodes)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
            &particleContributing)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::coordinates,
            (double const **) &coordinates);
    if (ier) {
        LOG_ERROR("Could not create model compute arguments input @ setDefaultInputs");
        return;
    }
    int const numberOfParticles = *numberOfParticlesPointer;

    torchModel->SetInputNode(0, particleContributing, numberOfParticles);
    torchModel->SetInputNode(1, coordinates, 3 * numberOfParticles, true);

    updateNeighborList(modelComputeArguments, numberOfParticles);
    torchModel->SetInputNode(2, num_neighbors_.data(), static_cast<int>(num_neighbors_.size()));
    torchModel->SetInputNode(3, neighbor_list.data(), static_cast<int>(neighbor_list.size()));
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriver::setDescriptorInputs(const KIM::ModelComputeArguments *modelComputeArguments) {
    int const *numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling
    int *particleContributing = nullptr;
    double *coordinates = nullptr;
    auto ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
            &numberOfParticlesPointer)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
            &particleSpeciesCodes)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
            &particleContributing)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::coordinates,
            (double const **) &coordinates);
    if (ier) {
        LOG_ERROR("Could not create model compute arguments input @ setDefaultInputs");
        return;
    }
    int neigh_from, n_neigh;
    neigh_from = 0;
    n_neigh = 0;

    int contributing_atoms_count = 0;
    for (int i = 0; i < *numberOfParticlesPointer; i++) {
        if (*(particleContributing + i) == 1) {
            contributing_atoms_count += 1;
        }
    }
    // Initialize descriptors on the basis of function
    auto option = torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true);

    // TODO this is temporary hard coded fix. Need to improve it by inheriting Base
    // descriptor class in all descriptors. **Priority**
    std::cout << "USING SYMUFUN WORKAROUND. FIX ME ASAP" << "\n";
    auto sf = reinterpret_cast<SymmetryFunctionParams *>(descriptor->descriptor_map["SymmetryFunction"]);
    int width = sf->width;
    //

    updateNeighborList(modelComputeArguments, *numberOfParticlesPointer);

    descriptor_array = new double[contributing_atoms_count * width];
    for (int i = 0; i < contributing_atoms_count; i++) {
        for (int j = i * width; j < (i + 1) * width; j++) { descriptor_array[j] = 0.; }
        n_neigh = num_neighbors_[i];
        std::vector<int> n_list(neighbor_list.begin() + neigh_from, neighbor_list.begin() + neigh_from + n_neigh);
        neigh_from += n_neigh;
        symmetry_function_atomic(i,
                                 coordinates,
                                 particleSpeciesCodes,
                                 n_list.data(),
                                 n_neigh,
                                 descriptor_array + (i * width),
                                 sf);
    }
    // auto descriptor_tensor = torch::from_blob(descriptor_array, {contributing_atoms_count, width}, option);

    std::vector<int> input_tensor_size({contributing_atoms_count, width});
    torchModel->SetInputNode(0, descriptor_array, input_tensor_size, true);
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriver::setGraphInputs(const KIM::ModelComputeArguments *modelComputeArguments) {
    int const *numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling
    int *particleContributing = nullptr;
    double *coordinates = nullptr;
    auto ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
            &numberOfParticlesPointer)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
            &particleSpeciesCodes)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
            &particleContributing)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::coordinates,
            (double const **) &coordinates);
    if (ier) {
        LOG_ERROR("Could not create model compute arguments input @ setDefaultInputs");
        return;
    }
    int numberOfNeighbors;
    int const *neighbors;
    std::tuple<int, int> bond_pair, rev_bond_pair;
    std::vector<std::set<std::tuple<int, int> > > unrolled_graph(n_layers);
    std::vector<int> next_list, prev_list;
    int contributing_atoms_count = 0;

    for (int i = 0; i < *numberOfParticlesPointer; i++) {
        if (*(particleContributing + i) == 1) {
            contributing_atoms_count += 1;
        }
    }

    for (int atom_i = 0; atom_i < contributing_atoms_count; atom_i++) {
        prev_list.push_back(atom_i);
        for (int i = 0; i < n_layers; i++) {
            std::set<std::tuple<int, int> > conv_layer;
            do {
                int curr_atom = prev_list.back();
                prev_list.pop_back();
                modelComputeArguments->GetNeighborList(0, curr_atom, &numberOfNeighbors, &neighbors);
                for (int j = 0; j < numberOfNeighbors; j++) {
                    bond_pair = std::make_tuple(curr_atom, neighbors[j]);
                    rev_bond_pair = std::make_tuple(neighbors[j], curr_atom);
                    conv_layer.insert(bond_pair);
                    conv_layer.insert(rev_bond_pair);
                    next_list.push_back((neighbors[j]));
                }
            } while (!prev_list.empty());
            prev_list.swap(next_list);
            unrolled_graph[i].insert(conv_layer.begin(), conv_layer.end());
        }
        prev_list.clear();
    }
    graph_set_to_graph_array(unrolled_graph);

    torchModel->SetInputNode(0, particleSpeciesCodes, *numberOfParticlesPointer, false);

    std::vector<int> input_tensor_size({*numberOfParticlesPointer, 3});
    torchModel->SetInputNode(1, coordinates, input_tensor_size, true);

    for (int i = 0; i < n_layers; i++) {
        torchModel->SetInputNode(2 + i, i, static_cast<int>(unrolled_graph[i].size()), graph_edge_indices);
    }
}

// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchMLModelDriver::readParameters(KIM::ModelDriverCreate *const modelDriverCreate,
                                        int *const ier) {
    // Read parameter files from model driver ---------------------------------------
    int num_param_files;
    std::string const *param_file_name, *tmp_file_name;
    std::string const *param_dir_name;
    std::string const *model_file_name;
    std::string const *descriptor_file_name;

    modelDriverCreate->GetNumberOfParameterFiles(&num_param_files);

    // Only 2 files expected .param and .pt
    if (num_param_files > MAX_FILE_NUM) {
        *ier = true;
        LOG_ERROR("Too many parameter files");
        return;
    }

    for (int i = 0; i < num_param_files; i++) {
        modelDriverCreate->GetParameterFileBasename(i, &tmp_file_name);
        if (tmp_file_name->substr(tmp_file_name->size() - 5) == "param") {
            param_file_name = tmp_file_name;
        } else if (tmp_file_name->substr(tmp_file_name->size() - 2) == "pt") {
            model_file_name = tmp_file_name;
        } else if (tmp_file_name->substr(tmp_file_name->size() - 3) == "dat") {
            descriptor_file_name = tmp_file_name;
        } else {
            LOG_ERROR("File extensions do not match; only expected .param or .pt");
            *ier = true;
            return;
        }
    }

    // Get param directory to load model and parameters from
    modelDriverCreate->GetParameterFileDirectoryName(&param_dir_name);

    std::string full_qualified_file_name = *param_dir_name + "/" + *param_file_name;
    std::string full_qualified_model_name = *param_dir_name + "/" + *model_file_name;

    std::string placeholder_string;

    std::fstream file_ptr(full_qualified_file_name);

    if (file_ptr.is_open()) {
        // TODO better structured input block. YAML?
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');

        n_elements = std::stoi(placeholder_string);
        std::getline(file_ptr, placeholder_string);

        for (int i = 0; i < n_elements; i++) {
            auto pos = placeholder_string.find(' ');
            elements_list.push_back(placeholder_string.substr(0, pos));
            if (pos == std::string::npos) {
                if (i + 1 != n_elements) {
                    LOG_ERROR("Incorrect formatting OR number of elements");
                    *ier = true;
                    return;
                }
                LOG_DEBUG("Number of elements read: " + std::to_string(i + 1));
            } else {
                placeholder_string.erase(0, pos + 1);
            }
        }
        // blank line
        std::getline(file_ptr, placeholder_string);
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');
        // which preprocessing to use
        preprocessing = placeholder_string;
        // std::transform(preprocessing.begin(),preprocessing.end(),
        //                preprocessing.begin(),::tolower);

        // blank line
        std::getline(file_ptr, placeholder_string);
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');
        // influence distance
        cutoff_distance = std::stod(placeholder_string);
        n_layers = 0;
        if (preprocessing == "Graph") {
            std::getline(file_ptr, placeholder_string);
            n_layers = std::stoi(placeholder_string);
            influence_distance = cutoff_distance * n_layers;
        } else {
            influence_distance = cutoff_distance;
        }

        // blank line
        std::getline(file_ptr, placeholder_string);
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');
        // Model name for comparison
        model_name = placeholder_string;

        // blank line
        std::getline(file_ptr, placeholder_string);
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');
        // Does the model return forces? If no then we need to compute gradients
        // If yes we can optimize it further using inference mode
        for (char &t: placeholder_string) t = static_cast<char>(tolower(t));
        returns_forces = placeholder_string == "true";

        // blank line
        std::getline(file_ptr, placeholder_string);
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');
        // number of strings
        number_of_inputs = std::stoi(placeholder_string);

        if (preprocessing == "Descriptor") {
            // blank line
            std::getline(file_ptr, placeholder_string);
            // Ignore comments
            do {
                std::getline(file_ptr, placeholder_string);
            } while (placeholder_string[0] == '#');
            // number of strings
            descriptor_name = placeholder_string;
            descriptor_param_file = *param_dir_name + "/" + *descriptor_file_name;
            // TODO raise exception for missing or unfound descriptor if needed?
        }

    } else {
        LOG_ERROR("Param file not found");
        *ier = true;
        return;
    }
    file_ptr.close();

    LOG_DEBUG("Successfully parsed parameter file");
    if (*model_file_name != model_name) {
        LOG_ERROR("Provided model file name different from present model file.");
        *ier = false;
        return;
    }
    // Load Torch Model ----------------------------------------------------------------
    torchModel = MLModel::create(full_qualified_model_name.c_str(),
                                 ML_MODEL_PYTORCH,
                                 std::getenv(KIM_DEVICE_ENV_VAR),
                                 number_of_inputs);
    LOG_INFORMATION("Loaded Torch model and set to eval");
}

// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchMLModelDriver::unitConversion(KIM::ModelDriverCreate *const modelDriverCreate,
                                        KIM::LengthUnit const requestedLengthUnit,
                                        KIM::EnergyUnit const requestedEnergyUnit,
                                        KIM::ChargeUnit const requestedChargeUnit,
                                        KIM::TemperatureUnit const requestedTemperatureUnit,
                                        KIM::TimeUnit const requestedTimeUnit,
                                        int *const ier) {
    KIM::LengthUnit fromLength = KIM::LENGTH_UNIT::A;
    KIM::EnergyUnit fromEnergy = KIM::ENERGY_UNIT::eV;
    KIM::ChargeUnit fromCharge = KIM::CHARGE_UNIT::e;
    KIM::TemperatureUnit fromTemperature = KIM::TEMPERATURE_UNIT::K;
    KIM::TimeUnit fromTime = KIM::TIME_UNIT::ps;
    double convertLength = 1.0;
    *ier = KIM::ModelDriverCreate::ConvertUnit(fromLength,
                                               fromEnergy,
                                               fromCharge,
                                               fromTemperature,
                                               fromTime,
                                               requestedLengthUnit,
                                               requestedEnergyUnit,
                                               requestedChargeUnit,
                                               requestedTemperatureUnit,
                                               requestedTimeUnit,
                                               1.0,
                                               0.0,
                                               0.0,
                                               0.0,
                                               0.0,
                                               &convertLength);

    if (*ier) {
        LOG_ERROR("Unable to convert length unit");
        return;
    }

    *ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                       requestedEnergyUnit,
                                       KIM::CHARGE_UNIT::unused,
                                       KIM::TEMPERATURE_UNIT::unused,
                                       KIM::TIME_UNIT::unused);

}

// --------------------------------------------------------------------------------
void TorchMLModelDriver::setSpecies(KIM::ModelDriverCreate *const modelDriverCreate, int *const ier) {
    for (auto const &species: elements_list) {
        KIM::SpeciesName const specName1(species);

        //    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator> modelSpeciesMap;
        //    std::vector<KIM::SpeciesName> speciesNameVector;
        //
        //    speciesNameVector.push_back(species);
        //    // check for new species
        //    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>::const_iterator iIter = modelSpeciesMap.find(specName1);
        // all the above is to remove species duplicates
        *ier = modelDriverCreate->SetSpeciesCode(specName1, 0);
        if (*ier) return;
    }
}

// --------------------------------------------------------------------------------
void TorchMLModelDriver::registerFunctionPointers(KIM::ModelDriverCreate *const modelDriverCreate, int *const ier) {
    // Use function pointer definitions to verify correct prototypes
    KIM::ModelDestroyFunction *destroy = TorchMLModelDriver::Destroy;
    KIM::ModelRefreshFunction *refresh = TorchMLModelDriver::Refresh;
    KIM::ModelComputeFunction *compute = TorchMLModelDriver::Compute;
    KIM::ModelComputeArgumentsCreateFunction *CACreate = TorchMLModelDriver::ComputeArgumentsCreate;
    KIM::ModelComputeArgumentsDestroyFunction *CADestroy = TorchMLModelDriver::ComputeArgumentsDestroy;

    *ier = modelDriverCreate->SetRoutinePointer(
            KIM::MODEL_ROUTINE_NAME::Destroy,
            KIM::LANGUAGE_NAME::cpp,
            true,
            reinterpret_cast<KIM::Function *>(destroy))
           || modelDriverCreate->SetRoutinePointer(
            KIM::MODEL_ROUTINE_NAME::Refresh,
            KIM::LANGUAGE_NAME::cpp,
            true,
            reinterpret_cast<KIM::Function *>(refresh))
           || modelDriverCreate->SetRoutinePointer(
            KIM::MODEL_ROUTINE_NAME::Compute,
            KIM::LANGUAGE_NAME::cpp,
            true,
            reinterpret_cast<KIM::Function *>(compute))
           || modelDriverCreate->SetRoutinePointer(
            KIM::MODEL_ROUTINE_NAME::ComputeArgumentsCreate,
            KIM::LANGUAGE_NAME::cpp,
            true,
            reinterpret_cast<KIM::Function *>(CACreate))
           || modelDriverCreate->SetRoutinePointer(
            KIM::MODEL_ROUTINE_NAME::ComputeArgumentsDestroy,
            KIM::LANGUAGE_NAME::cpp,
            true,
            reinterpret_cast<KIM::Function *>(CADestroy));
}

//******************************************************************************
// static member function
int TorchMLModelDriver::ComputeArgumentsDestroy(
        KIM::ModelCompute const *const modelCompute,
        KIM::ModelComputeArgumentsDestroy *const modelComputeArgumentsDestroy) {
    return false;
}

// ---------------------------------------------------------------------------------
void TorchMLModelDriver::graph_set_to_graph_array(std::vector<std::set<std::tuple<int, int>>> &
unrolled_graph) {
    int i = 0;
    for (auto const edge_index_set: unrolled_graph) {
        int j = 0;
        int graph_size = static_cast<int>(edge_index_set.size());
        graph_edge_indices[i] = new int[graph_size * 2];
        for (auto bond_pair: edge_index_set) {
            graph_edge_indices[i][j] = std::get<0>(bond_pair);
            graph_edge_indices[i][j + graph_size] = std::get<1>(bond_pair);
            j++;
        }
        i++;
    }
}


// *****************************************************************************
TorchMLModelDriver::~TorchMLModelDriver() {
    delete descriptor_array;
    if (preprocessing == "Graph") {
        for (int i = 0; i < n_layers; i++) delete graph_edge_indices[i];
    }
    delete graph_edge_indices;
}