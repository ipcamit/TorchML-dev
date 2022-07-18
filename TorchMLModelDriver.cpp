#include "TorchMLModelDriver.hpp"
//#include "TorchMLModelImplementation.hpp"
//#include <cstddef>
#include <iostream>
#include "KIM_LogMacros.hpp"
#include <fstream>
#include <map>
#include <vector>

#define MAX_FILE_NUM 2
typedef double VecOfSize3[3];
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
    TorchMLModelDriver *const modelObject
            = new TorchMLModelDriver(modelDriverCreate,
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
    *ier = false;

    // Read parameter files from model driver ---------------------------------------
    readParameters(modelDriverCreate, ier);
    LOG_DEBUG("Read Param files");
    if (*ier) return;

    // Unit conversions -----------------------------------------------------------------
    unitConversion(modelDriverCreate, requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
                   requestedTemperatureUnit, requestedTimeUnit, ier);
    LOG_DEBUG("Registered Unit Conversion");
    if (*ier) return;

    // Set Influence distance ---------------------------------------------------------
    int modelWillNotRequestNeighborsOfNoncontributingParticles_ = 1;
    modelDriverCreate->SetInfluenceDistancePointer(&influence_distance);
    modelDriverCreate->SetNeighborListPointers(1, &influence_distance,
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

    return;
}

//******************************************************************************
//TorchMLModelDriver::~TorchMLModelDriver() { delete implementation_; }

//******************************************************************************
// static member function
int TorchMLModelDriver::Destroy(KIM::ModelDestroy *const modelDestroy) {
    TorchMLModelDriver *modelObject;
    modelDestroy->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

    if (modelObject != NULL) {
        // delete object itself
        delete modelObject;
    }

    // everything is good
    return false;
}

//******************************************************************************
// static member function
int TorchMLModelDriver::Refresh(KIM::ModelRefresh *const modelRefresh) {
    TorchMLModelDriver *modelObject;
    modelRefresh->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
//    modelObject->m = 0.0;
//    modelObject->c = 0.0;
    return false;
}

//******************************************************************************
// static member function
int TorchMLModelDriver::Compute(
        KIM::ModelCompute const *const modelCompute,
        KIM::ModelComputeArguments const *const modelComputeArguments) {

    TorchMLModelDriver *modelObject;
    modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

    int const *numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling
    int *particleContributing = NULL;
    double *forces = NULL;
    double *coordinates = NULL;
    VecOfSize3 *coordinates3;
    double *energy = NULL;
    int numOfNeighbors;
    int const *neighbors;

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

    int const numberOfParticles = *numberOfParticlesPointer;

    modelObject->torchModel->SetInputNode(0, particleContributing, numberOfParticles);
    modelObject->torchModel->SetInputNode(1, coordinates, 3 * numberOfParticles, true);

    for (int i = 0; i < numberOfParticles; i++) {
        modelComputeArguments->GetNeighborList(0,
                                               i,
                                               &numOfNeighbors,
                                               &neighbors);
        modelObject->num_neighbors_.push_back(numOfNeighbors);
        for (int neigh = 0; neigh < numOfNeighbors; neigh++) {
            modelObject->neighbor_list.push_back(neighbors[neigh]);
        }
    }

    modelObject->torchModel->SetInputNode(2, modelObject->num_neighbors_.data(), modelObject->num_neighbors_.size());
    modelObject->torchModel->SetInputNode(3, modelObject->neighbor_list.data(), modelObject->neighbor_list.size());
    modelObject->torchModel->Run(energy, forces);

    return ier;
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

// Auxiliary methods------------------------------------------------------------
//void TorchMLModelDriver::updateNeighborList() {
//
//}
//
//void TorchMLModelDriver::setInputs(TorchMLModelDriver & modelObject) {
//
//}
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchMLModelDriver::readParameters(KIM::ModelDriverCreate *const modelDriverCreate,
                                        int *const ier) {
    // Read parameter files from model driver ---------------------------------------
    int num_param_files;
    std::string const *param_file_name, *tmp_file_name;
    std::string const *param_dir_name;
    std::string const *model_file_name;

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
            auto pos = placeholder_string.find(" ");
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

        // blank line
        std::getline(file_ptr, placeholder_string);
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');
        // which preprocessing to use
        influence_distance = std::stod(placeholder_string);

        // blank line
        std::getline(file_ptr, placeholder_string);
        // Ignore comments
        do {
            std::getline(file_ptr, placeholder_string);
        } while (placeholder_string[0] == '#');

        model_name = placeholder_string;
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
                                 std::getenv(KIM_DEVICE_ENV_VAR));
    LOG_INFORMATION("Loaded Torch model and set to eval");
}

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

void TorchMLModelDriver::setSpecies(KIM::ModelDriverCreate *const modelDriverCreate, int * const ier) {
        for (auto species: elements_list) {
        KIM::SpeciesName const specName1(species);

        //    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator> modelSpeciesMap;
        //    std::vector<KIM::SpeciesName> speciesNameVector;
        //
        //    speciesNameVector.push_back(species);
        //    // check for new species
        //    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>::const_iterator iIter = modelSpeciesMap.find(specName1);
        // all of the above is to remove species duplicates
        *ier = modelDriverCreate->SetSpeciesCode(specName1, 0);
        if (*ier) return;
    }
}

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