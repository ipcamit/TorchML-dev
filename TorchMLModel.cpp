#include "TorchMLModel.hpp"
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
int model_driver_create(KIM::ModelDriverCreate * const modelDriverCreate,
                        KIM::LengthUnit const requestedLengthUnit,
                        KIM::EnergyUnit const requestedEnergyUnit,
                        KIM::ChargeUnit const requestedChargeUnit,
                        KIM::TemperatureUnit const requestedTemperatureUnit,
                        KIM::TimeUnit const requestedTimeUnit)
{
  int ier;
  // read input files, convert units if needed, compute
  // interpolation coefficients, set cutoff, and publish parameters
  TorchMLModel * const modelObject
      = new TorchMLModel(modelDriverCreate,
                            requestedLengthUnit,
                            requestedEnergyUnit,
                            requestedChargeUnit,
                            requestedTemperatureUnit,
                            requestedTimeUnit,
                            &ier);

  if (ier)
  {
    // constructor already reported the error
    delete modelObject;
    return ier;
  }

  // register pointer to TorchMLModel object in KIM object
  modelDriverCreate->SetModelBufferPointer(static_cast<void *>(modelObject));

  // everything is good
  ier = false;
  return ier;
}
}  // extern "C"

//==============================================================================
//
// Implementation of TorchMLModel public wrapper functions
//
//==============================================================================

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
TorchMLModel::TorchMLModel(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
    *ier = false;

    // Read parameter files from model driver ---------------------------------------
    int numberParameterFiles;
    modelDriverCreate->GetNumberOfParameterFiles(&numberParameterFiles);
    std::string const * paramFileName, * tmpFileName;
    std::string const * paramDirectory;
    std::string const * modelFileName;

    // Only 2 files expected .param and .pt
    std::cout<<"Param files count: " << numberParameterFiles <<"\n";
    if (numberParameterFiles > MAX_FILE_NUM){
        *ier = true;
        LOG_ERROR("Too many parameter files");
        return;
    }

    for (int i = 0; i < numberParameterFiles; i++){
        modelDriverCreate->GetParameterFileBasename(i,&tmpFileName);
        if (tmpFileName->substr(tmpFileName->size() - 5) == "param"){
            paramFileName = tmpFileName;
        }
        else if (tmpFileName->substr(tmpFileName->size() - 2) == "pt"){
            modelFileName = tmpFileName;
        } else {
            LOG_ERROR("File extensions do not match; only expected .param or .pt");
            *ier = true;
            return;
        }
    }

    // Get param directory to load model and parameters from
    modelDriverCreate->GetParameterFileDirectoryName(&paramDirectory);

    auto fullyQualifiedParamFileName = *paramDirectory + "/" + *paramFileName;
    auto fullyQualifiedModelName = *paramDirectory + "/" + *modelFileName;
    std::ifstream filePtr(fullyQualifiedParamFileName);
    std::string dummyStr;
    if (filePtr.is_open()){
        // TODO better structured input block. YAML?
        // Comments if needed
        do {
            std::getline(filePtr, dummyStr);
        } while (dummyStr[0] == '#');
        n_elements = std::stoi(dummyStr);
        std::getline(filePtr, dummyStr);
        int pos;
        for (int i = 0; i < n_elements ; i++){
            pos = dummyStr.find(" ");
            elements_list.push_back(dummyStr.substr(0,pos));
            if (pos == std::string::npos){
                if (i + 1 != n_elements){
                    LOG_ERROR("Incorrect formatting OR number of elements");
                }
                LOG_INFORMATION("Number of elements read: " + std::to_string(i+1));
            } else {
                dummyStr.erase(0,pos + 1);
            }
        }
        // blank line
        std::getline(filePtr, dummyStr);
        // Ignore comments
        do {
            std::getline(filePtr, dummyStr);
        } while (dummyStr[0] == '#');

        preprocessing = dummyStr;
        std::cout << n_elements << preprocessing<<"\n";

        // blank line
        std::getline(filePtr, dummyStr);
        // Ignore comments
        do {
            std::getline(filePtr, dummyStr);
        } while (dummyStr[0] == '#');

        model_name =  dummyStr;
        std::cout << model_name<<"\n";
    } else {
        LOG_ERROR("Param file not found");
        *ier = true;
        return;
    }
    filePtr.close();
    LOG_INFORMATION("Successfully parsed parameter file");
    if (*modelFileName != model_name){
        LOG_ERROR("Provided model file name different from present model file.");
        *ier = false;
        return;
    }
    std::cout <<"read files\n";

    // Load Torch Model ----------------------------------------------------------------
    TorchModel = MLModel::create(fullyQualifiedModelName.c_str(),
                                 ML_MODEL_PYTORCH,
                                 std::getenv(KIM_DEVICE_ENV_VAR));
    LOG_INFORMATION("Loaded Torch model and set to eval");

    // Unit conversions -----------------------------------------------------------------
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
    std::cout << "Conversion done\n";

    *ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                    requestedEnergyUnit,
                                    KIM::CHARGE_UNIT::unused,
                                    KIM::TEMPERATURE_UNIT::unused,
                                    KIM::TIME_UNIT::unused);

    std::cout << "Units set\n";

    // Set Influeance distance ---------------------------------------------------------
    inflDist = 3.77118;
    int modelWillNotRequestNeighborsOfNoncontributingParticles_ = 1;
    modelDriverCreate->SetInfluenceDistancePointer(&inflDist);
    modelDriverCreate->SetNeighborListPointers(1, &inflDist,
                                               &modelWillNotRequestNeighborsOfNoncontributingParticles_);

    // Species code --------------------------------------------------------------------
    for (auto species : elements_list) {
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

    // Register Index settings-----------------------------------------------------------
    modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);
    std::cout << "indexing done\n";

    // Register Parameters --------------------------------------------------------------
    // Currently trying to create a simple Linear Model with two parameters from param file
    //Extent = number of entries in parameter pointer
    // All model parameters are inside model So nothing to do here?
    *ier = modelDriverCreate->SetParameterPointer(1,&n_elements, "n_elements", "slope");
    if (*ier) { LOG_ERROR("parameter m"); return;}
//    *ier = modelDriverCreate->SetParameterPointer(1,&c, "c", "slope");
//    if (*ier) { LOG_ERROR("parameter c"); return;}
    std::cout << "parameters done\n";

    // register funciton pointers -----------------------------------------------------------
    // Use function pointer definitions to verify correct prototypes
    KIM::ModelDestroyFunction * destroy = TorchMLModel::Destroy;
    KIM::ModelRefreshFunction * refresh = TorchMLModel::Refresh;
    KIM::ModelComputeFunction * compute = TorchMLModel::Compute;
    KIM::ModelComputeArgumentsCreateFunction * CACreate = TorchMLModel::ComputeArgumentsCreate;
    KIM::ModelComputeArgumentsDestroyFunction * CADestroy = TorchMLModel::ComputeArgumentsDestroy;

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
    if(*ier) {
        return;
    }

    std::cout<<"Registered function pointers\n";

    *ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                    requestedEnergyUnit,
                                    KIM::CHARGE_UNIT::unused,
                                    KIM::TEMPERATURE_UNIT::unused,
                                    KIM::TIME_UNIT::unused);

    return;
    }

//******************************************************************************
//TorchMLModel::~TorchMLModel() { delete implementation_; }

//******************************************************************************
// static member function
int TorchMLModel::Destroy(KIM::ModelDestroy * const modelDestroy)
{
  TorchMLModel * modelObject;
  modelDestroy->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

  if (modelObject != NULL)
  {
    // delete object itself
    delete modelObject;
  }

  // everything is good
  return false;
}

//******************************************************************************
// static member function
int TorchMLModel::Refresh(KIM::ModelRefresh * const modelRefresh){
    TorchMLModel * modelObject;
    modelRefresh->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
//    modelObject->m = 0.0;
//    modelObject->c = 0.0;
    return false;
}

//******************************************************************************
// static member function
int TorchMLModel::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments){

    TorchMLModel * modelObject;
    modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

//    auto m_ = modelObject->m;
//    auto c_ = modelObject->c;

    int const * numberOfParticlesPointer;
    int *particleSpeciesCodes; // FIXME: Implement species code handling
    int *particleContributing = NULL;
    double * forces = NULL;
    double * coordinates = NULL;
    VecOfSize3 * coordinates3;
    double * energy = NULL;
    std::vector<int> num_neighbors_;
    std::vector<int> neighbor_list;
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

    modelObject->TorchModel->SetInputNode(0, particleContributing, numberOfParticles);
    modelObject->TorchModel->SetInputNode(1, coordinates, 3 * numberOfParticles, true);

    for (int i = 0; i< numberOfParticles; i++){
        modelComputeArguments->GetNeighborList(0,
                                               i,
                                               &numOfNeighbors,
                                               &neighbors);
        num_neighbors_.push_back(numOfNeighbors);
        for (int neigh = 0; neigh < numOfNeighbors; neigh++){
            neighbor_list.push_back(neighbors[neigh]);
        }
    }

    modelObject->TorchModel->SetInputNode(2, num_neighbors_.data(), num_neighbors_.size());
    modelObject->TorchModel->SetInputNode(3, neighbor_list.data(), neighbor_list.size());
    modelObject->TorchModel->Run(energy, forces);

    return ier;
}

//******************************************************************************
// static member function
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate
int TorchMLModel::ComputeArgumentsCreate(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate){
//    TorchMLModel * modelObject;
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

//******************************************************************************
// static member function
int TorchMLModel::ComputeArgumentsDestroy(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy)
{
    return false;
}