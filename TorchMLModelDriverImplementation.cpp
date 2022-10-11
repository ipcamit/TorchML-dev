#include "TorchMLModelDriverImplementation.hpp"
#include "TorchMLModelDriver.hpp"
#include "KIM_LogMacros.hpp"
#include <map>
#include <vector>
#include <algorithm>
#include <stdexcept>

#define MAX_FILE_NUM 3
//typedef double VecOfSize3[3];
#define KIM_DEVICE_ENV_VAR "KIM_MODEL_EXECUTION_DEVICE"

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

TorchMLModelDriverImplementation::TorchMLModelDriverImplementation(
        KIM::ModelDriverCreate *const modelDriverCreate,
        KIM::LengthUnit const requestedLengthUnit,
        KIM::EnergyUnit const requestedEnergyUnit,
        KIM::ChargeUnit const requestedChargeUnit,
        KIM::TemperatureUnit const requestedTemperatureUnit,
        KIM::TimeUnit const requestedTimeUnit,
        int *const ier) {
    *ier = false;
    //initialize members to remove warning----
    influence_distance = 0.0;
    n_elements = 0;
    mlModel = nullptr;
    returns_forces = false;
    cutoff_distance = 0.0;
    n_layers = 0;
    n_contributing_atoms = 0;
    number_of_inputs = 0;
    // Read parameter files from model driver ---------------------------------------
    // also initialize the mlModel
    readParametersFile(modelDriverCreate, ier);
    LOG_DEBUG("Read Param files");
    if (*ier) return;

    // Unit conversions -----------------------------------------------------------------
    unitConversion(modelDriverCreate, requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
                   requestedTemperatureUnit, requestedTimeUnit, ier);
    LOG_DEBUG("Registered Unit Conversion");
    if (*ier) return;

    // Set Influence distance ---------------------------------------------------------
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
        descriptor = DescriptorKind::initDescriptor(descriptor_param_file, descriptor_kind);
        graph_edge_indices = nullptr;
    } else if (preprocessing == "Graph") {
        graph_edge_indices = new long *[n_layers];
        for (int i = 0; i < n_layers; i++) graph_edge_indices[i] = nullptr;
        descriptor = nullptr;
    } else {
        descriptor = nullptr;
        graph_edge_indices = nullptr;
    }
    descriptor_array = nullptr;
    species_atomic_number = nullptr;
    contraction_array = nullptr;
    std::cout << "Initialized\n";

}

//******************************************************************************
int TorchMLModelDriverImplementation::Refresh(KIM::ModelRefresh *const modelRefresh) {
    TorchMLModelDriver *modelObject; //To silence the compiler
    modelRefresh->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
    // As all param are part of torch model, nothing to do here?
    // TODO Distance matrix for computational efficiency, which will be refreshed to -1
    return false;
}

//******************************************************************************
int TorchMLModelDriverImplementation::Compute(
        KIM::ModelComputeArguments const *const modelComputeArguments) {

    Run(modelComputeArguments);
    // TODO see proper way to return error codes
    return false;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate

int TorchMLModelDriverImplementation::ComputeArgumentsCreate(
        KIM::ModelComputeArgumentsCreate *const modelComputeArgumentsCreate) {
    LOG_INFORMATION("Compute argument create");
    int error = modelComputeArgumentsCreate->SetArgumentSupportStatus(
            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
            KIM::SUPPORT_STATUS::required)
                    // Support state can be optional for energy
                    // But then we need to explicitly handle nullptr case
                    // As energy is anyway always needs to be computed
                    // as a prerequisite for force computation, easier to
                    // make it required.
                    // TODO: Handle it properly in future for GD models
                || modelComputeArgumentsCreate->SetArgumentSupportStatus(
            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
            KIM::SUPPORT_STATUS::optional);
    // register callbacks
    LOG_INFORMATION("Register callback supportStatus");
    // error = error
    //       || modelComputeArgumentsCreate->SetCallbackSupportStatus(
    //           KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
    //           KIM::SUPPORT_STATUS::optional)
    //       || modelComputeArgumentsCreate->SetCallbackSupportStatus(
    //           KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
    //           KIM::SUPPORT_STATUS::optional);
    return error;
}

// *****************************************************************************
// Auxiliary methods------------------------------------------------------------

void TorchMLModelDriverImplementation::Run(const KIM::ModelComputeArguments *const modelComputeArguments) {
    c10::IValue out_tensor;
    contributingAtomCounts(modelComputeArguments);
    preprocessInputs(modelComputeArguments);
    mlModel->Run(out_tensor);
    postprocessOutputs(out_tensor, modelComputeArguments);
}

// -----------------------------------------------------------------------------
void TorchMLModelDriverImplementation::preprocessInputs(KIM::ModelComputeArguments const *const modelComputeArguments) {
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
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::postprocessOutputs(c10::IValue &out_tensor,
                                                          KIM::ModelComputeArguments const *modelComputeArguments) {

    double *energy = nullptr;
    double *forces = nullptr;
    int const *numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling
    double *coordinates = nullptr;

    auto ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
            &numberOfParticlesPointer)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
            &particleSpeciesCodes)
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

    if (returns_forces) {
        const auto output_tensor_list = out_tensor.toTuple()->elements();
        *energy = *output_tensor_list[0].toTensor().to(torch::kCPU).data_ptr<double>();
        auto torch_forces = output_tensor_list[1].toTensor().to(torch::kCPU);
        auto force_accessor = torch_forces.accessor<double, 1>();
        for (int atom_count = 0; atom_count < force_accessor.size(0); ++atom_count) {
            forces[atom_count] = -force_accessor[atom_count];
        }
    } else {
        // TODO: partial particle energy
        // sum() leaves scalars intact, therefore can be used here
        // but in future scalar and vector tensors will be handled differently
        // to ensure proper handling of partial particle energy parameter from KIM
        out_tensor.toTensor().sum().backward();
        std::cout << "Took Backwards\n";
        std::cout << out_tensor.toTensor().sum();
        c10::IValue input_tensor;
        mlModel->GetInputNode(input_tensor);
        auto input_grad = input_tensor.toTensor().grad();
//        std::cout << input_grad;
        *energy = *out_tensor.toTensor().to(torch::kCPU).data_ptr<double>();
        int neigh_from = 0;
        int n_neigh;
        if (preprocessing == "Descriptor") {
            int width = descriptor->width;
            std::cout << "Width in post " << width <<"\n";
            for (int i = 0; i < n_contributing_atoms; i++) {
                n_neigh = num_neighbors_[i];
                std::vector<int> n_list(neighbor_list.begin() + neigh_from,
                                        neighbor_list.begin() + neigh_from + n_neigh);
                neigh_from += n_neigh;
                // Single atom gradient from descriptor
                // TODO: call gradient function, which handles atom-wise iteration
                std::cout << "Grad: " << i << "\n";
                gradient_single_atom(i,
                                     *particleSpeciesCodes,
                                     particleSpeciesCodes,
                                     n_list.data(),
                                     n_neigh,
                                     coordinates,
                                     forces,
                                     input_tensor.toTensor().data_ptr<double>() + (i * width),
                                     input_grad.data_ptr<double>() + (i * width),
                                     descriptor);

            }
            std::cout << "Computed gradients\n";
            for (int i = 0; i < *numberOfParticlesPointer; i++) {
                // forces = -grad
                *(forces + 3 * i + 0) *= -1.0;
                *(forces + 3 * i + 1) *= -1.0;
                *(forces + 3 * i + 2) *= -1.0;
            }
        } else {
            auto force_accessor = input_grad.accessor<double, 2>();
            for (int i = 0; i < force_accessor.size(0); ++i) {
                *(forces + 3 * i + 0) = -force_accessor[i][0];
                *(forces + 3 * i + 1) = -force_accessor[i][1];
                *(forces + 3 * i + 2) = -force_accessor[i][2];
            }
        }
    }
}

// -----------------------------------------------------------------------------
void TorchMLModelDriverImplementation::updateNeighborList(KIM::ModelComputeArguments const *const modelComputeArguments,
                                                          int const numberOfParticles) {
    int numOfNeighbors;
    int const *neighbors;
    num_neighbors_.clear();
    neighbor_list.clear();
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

void TorchMLModelDriverImplementation::setDefaultInputs(const KIM::ModelComputeArguments *modelComputeArguments) {
    int const *numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling Ask Ryan
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

    mlModel->SetInputNode(0, particleContributing, numberOfParticles);
    mlModel->SetInputNode(1, coordinates, 3 * numberOfParticles, true);

    updateNeighborList(modelComputeArguments, n_contributing_atoms);
    mlModel->SetInputNode(2, num_neighbors_.data(), static_cast<int>(num_neighbors_.size()));
    mlModel->SetInputNode(3, neighbor_list.data(), static_cast<int>(neighbor_list.size()));
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::setDescriptorInputs(const KIM::ModelComputeArguments *modelComputeArguments) {
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
    int width = descriptor->width;
    std::cout << "Accessed width\n";
    updateNeighborList(modelComputeArguments, n_contributing_atoms);
    std::cout << "updated neighbors\n";
    if (descriptor_array) {
        delete[] descriptor_array;
        descriptor_array = nullptr;
    }
    std::cout << "Erased prev allocation\n";
    std::cout << n_contributing_atoms <<"   " << width <<"\n";
    descriptor_array = new double[n_contributing_atoms * width];
    std::cout << "allocated descriptor\n";
    for (int i = 0; i < n_contributing_atoms; i++) {
        for (int j = i * width; j < (i + 1) * width; j++) { descriptor_array[j] = 0.; }
        n_neigh = num_neighbors_[i];
        std::vector<int> n_list(neighbor_list.begin() + neigh_from, neighbor_list.begin() + neigh_from + n_neigh);
        neigh_from += n_neigh;
        // Single atom descriptor wrapper from descriptor
        // TODO: call compute function, which handles atom-wise iteration
//        std::cout << "i: " << i <<"\n";
        compute_single_atom(i,
                            *numberOfParticlesPointer,
                            particleSpeciesCodes,
                            n_list.data(),
                            n_neigh,
                            coordinates,
                            descriptor_array + (i * width),
                            descriptor);
    }
    std::cout << "Computed Descriptor\n";
    // auto descriptor_tensor = torch::from_blob(descriptor_array, {n_contributing_atoms, width}, option);

    std::vector<int> input_tensor_size({n_contributing_atoms, width});
    mlModel->SetInputNode(0, descriptor_array, input_tensor_size, true);
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::setGraphInputs(const KIM::ModelComputeArguments *modelComputeArguments) {
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
    std::vector<std::set<std::tuple<long, long> > > unrolled_graph(n_layers);
    std::vector<int> next_list, prev_list;

    double cutoff_sq = cutoff_distance * cutoff_distance;

    for (int atom_i = 0; atom_i < n_contributing_atoms; atom_i++) {
        prev_list.push_back(atom_i);
        for (int i = 0; i < n_layers; i++) {
            if (!prev_list.empty()) {
                do {
                    int curr_atom = prev_list.back();
                    prev_list.pop_back();
                    modelComputeArguments->GetNeighborList(0,
                                                           curr_atom,
                                                           &numberOfNeighbors,
                                                           &neighbors);
                    for (int j = 0; j < numberOfNeighbors; j++) {
                        if ((std::pow((*(coordinates + 3 * curr_atom + 0) - *(coordinates + 3 * neighbors[j] + 0)), 2) +
                             std::pow((*(coordinates + 3 * curr_atom + 1) - *(coordinates + 3 * neighbors[j] + 1)), 2) +
                             std::pow((*(coordinates + 3 * curr_atom + 2) - *(coordinates + 3 * neighbors[j] + 2)), 2))
                            <= cutoff_sq) {
                            bond_pair = std::make_tuple(curr_atom, neighbors[j]);
                            rev_bond_pair = std::make_tuple(neighbors[j], curr_atom);
                            unrolled_graph[i].insert(bond_pair);
                            unrolled_graph[i].insert(rev_bond_pair);
                            next_list.push_back((neighbors[j]));
                        }
                    }
                } while (!prev_list.empty());
                prev_list.swap(next_list);
            }
        }
        prev_list.clear();
    }
    graphSetToGraphArray(unrolled_graph);

    if (species_atomic_number) {
        delete[] species_atomic_number;
        species_atomic_number = nullptr;
    }

    species_atomic_number = new int[*numberOfParticlesPointer];
    for (int i = 0; i < *numberOfParticlesPointer; i++) {
        species_atomic_number[i] = z_map[particleSpeciesCodes[i]];
    }


    mlModel->SetInputNode(0, species_atomic_number, *numberOfParticlesPointer, false);

    std::vector<int> input_tensor_size({*numberOfParticlesPointer, 3});
    mlModel->SetInputNode(1, coordinates, input_tensor_size, true);

    for (int i = 0; i < n_layers; i++) {
        mlModel->SetInputNode(2 + i, i, static_cast<int>(unrolled_graph[i].size()), graph_edge_indices);
    }

    if (contraction_array) {
        delete[] contraction_array;
        contraction_array = nullptr;
    }

    contraction_array = new int64_t[*numberOfParticlesPointer];
    for (int i = 0; i < *numberOfParticlesPointer; i++) {
        contraction_array[i] = (i < n_contributing_atoms) ? 0 : 1;
    }
    mlModel->SetInputNode(2 + n_layers, contraction_array, *numberOfParticlesPointer, false);
}

// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchMLModelDriverImplementation::readParametersFile(KIM::ModelDriverCreate *modelDriverCreate,
                                                          int *ier) {
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
        // Species Z map
        for (int i = 0; i < n_elements; i++) {
            z_map.push_back(sym_to_z(elements_list[i]));
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
            if (descriptor_name == "SymmetryFunctions"){
                descriptor_kind = AvailableDescriptor::KindSymmetryFunctions;
            } else if (descriptor_name == "Bispectrum"){
                descriptor_kind = AvailableDescriptor::KindBispectrum;
            } else {
                throw std::invalid_argument("Descriptor not supported.");
            }
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
    mlModel = MLModel::create(full_qualified_model_name.c_str(),
                              ML_MODEL_PYTORCH,
                              std::getenv(KIM_DEVICE_ENV_VAR),
                              number_of_inputs);
    LOG_INFORMATION("Loaded Torch model and set to eval");
}

// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchMLModelDriverImplementation::unitConversion(KIM::ModelDriverCreate *const modelDriverCreate,
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
void TorchMLModelDriverImplementation::setSpecies(KIM::ModelDriverCreate *const modelDriverCreate, int *const ier) {
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
void TorchMLModelDriverImplementation::registerFunctionPointers(KIM::ModelDriverCreate *const modelDriverCreate,
                                                                int *const ier) {
    // Use function pointer definitions to verify correct prototypes
    // TODO This doesn't look nice, implementation calling parent class
    // See if there is a workaround
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
int TorchMLModelDriverImplementation::ComputeArgumentsDestroy(
        KIM::ModelComputeArgumentsDestroy *const modelComputeArgumentsDestroy) {
    // Nothing to do here?
    TorchMLModelDriver *modelObject; // To silence the compiler
    modelComputeArgumentsDestroy->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
    return false;
}

// ---------------------------------------------------------------------------------
void
TorchMLModelDriverImplementation::graphSetToGraphArray(std::vector<std::set<std::tuple<long, long>>> &unrolled_graph) {


    int i = 0;
    for (auto const &edge_index_set: unrolled_graph) {
        int j = 0;
        int graph_size = static_cast<int>(edge_index_set.size());
        // Sanitize previous graph
        if (graph_edge_indices[i]) {
            delete[] graph_edge_indices[i];
            graph_edge_indices[i] = nullptr;
        }
        graph_edge_indices[i] = new long[graph_size * 2];
        for (auto bond_pair: edge_index_set) {
            graph_edge_indices[i][j] = std::get<0>(bond_pair);
            graph_edge_indices[i][j + graph_size] = std::get<1>(bond_pair);
            j++;
        }
        i++;
    }
}

//-------------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::contributingAtomCounts(const KIM::ModelComputeArguments *modelComputeArguments) {
    int *numberOfParticlesPointer;
    int *particleContributing;
    auto ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
            &numberOfParticlesPointer)
               || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
            &particleContributing);
    if (ier) {
        LOG_ERROR("Could not get number of particles @ contributingAtomCount");
        return;
    }
    n_contributing_atoms = 0;
    for (int i = 0; i < *numberOfParticlesPointer; i++) {
        if (particleContributing[i] == 1) {
            n_contributing_atoms += 1;
        }
    }
}


// *****************************************************************************
TorchMLModelDriverImplementation::~TorchMLModelDriverImplementation() {
    delete[] descriptor_array;
    if (preprocessing == "Graph") {
        for (int i = 0; i < n_layers; i++) delete[] graph_edge_indices[i];
    }
    delete[] graph_edge_indices;
    delete descriptor;
    delete[] species_atomic_number;
    delete[] contraction_array;
}

// *****************************************************************************

int sym_to_z(std::string &sym) {
    // TODO more idiomatic handling of species. Ask Ryan
    if (sym == "H") return 1;
    if (sym == "He") return 2;
    if (sym == "Li") return 3;
    if (sym == "Be") return 4;
    if (sym == "B") return 5;
    if (sym == "C") return 6;
    if (sym == "N") return 7;
    if (sym == "O") return 8;
    if (sym == "F") return 9;
    if (sym == "Ne") return 10;
    if (sym == "Na") return 11;
    if (sym == "Mg") return 12;
    if (sym == "Al") return 13;
    if (sym == "Si") return 14;
    if (sym == "P") return 15;
    if (sym == "S") return 16;
    if (sym == "Cl") return 17;
    if (sym == "A") return 18;
    if (sym == "K") return 19;
    if (sym == "Ca") return 20;
    if (sym == "Sc") return 21;
    if (sym == "Ti") return 22;
    if (sym == "V") return 23;
    if (sym == "Cr") return 24;
    if (sym == "Mn") return 25;
    if (sym == "Fe") return 26;
    if (sym == "Co") return 27;
    if (sym == "Ni") return 28;
    if (sym == "Cu") return 29;
    if (sym == "Zn") return 30;
    if (sym == "Ga") return 31;
    if (sym == "Ge") return 32;
    if (sym == "As") return 33;
    if (sym == "Se") return 34;
    if (sym == "Br") return 35;
    if (sym == "Kr") return 36;
    if (sym == "Rb") return 37;
    if (sym == "Sr") return 38;
    if (sym == "Y") return 39;
    if (sym == "Zr") return 40;
    if (sym == "Nb") return 41;
    if (sym == "Mo") return 42;
    if (sym == "Tc") return 43;
    if (sym == "Ru") return 44;
    if (sym == "Rh") return 45;
    if (sym == "Pd") return 46;
    if (sym == "Ag") return 47;
    if (sym == "Cd") return 48;
    if (sym == "In") return 49;
    if (sym == "Sn") return 50;
    if (sym == "Sb") return 51;
    if (sym == "Te") return 52;
    if (sym == "I") return 53;
    if (sym == "Xe") return 54;
    if (sym == "Cs") return 55;
    if (sym == "Ba") return 56;
    if (sym == "La") return 57;
    if (sym == "Ce") return 58;
    if (sym == "Pr") return 59;
    if (sym == "Nd") return 60;
    if (sym == "Pm") return 61;
    if (sym == "Sm") return 62;
    if (sym == "Eu") return 63;
    if (sym == "Gd") return 64;
    if (sym == "Tb") return 65;
    if (sym == "Dy") return 66;
    if (sym == "Ho") return 67;
    if (sym == "Er") return 68;
    if (sym == "Tm") return 69;
    if (sym == "Yb") return 70;
    if (sym == "Lu") return 71;
    if (sym == "Hf") return 72;
    if (sym == "Ta") return 73;
    if (sym == "W") return 74;
    if (sym == "Re") return 75;
    if (sym == "Os") return 76;
    if (sym == "Ir") return 77;
    if (sym == "Pt") return 78;
    if (sym == "Au") return 79;
    if (sym == "Hg") return 80;
    if (sym == "Ti") return 81;
    if (sym == "Pb") return 82;
    if (sym == "Bi") return 83;
    if (sym == "Po") return 84;
    if (sym == "At") return 85;
    if (sym == "Rn") return 86;
    if (sym == "Fr") return 87;
    if (sym == "Ra") return 88;
    if (sym == "Ac") return 89;
    if (sym == "Th") return 90;
    if (sym == "Pa") return 91;
    if (sym == "U") return 92;
}