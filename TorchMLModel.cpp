#include "TorchMLModel.hpp"
//#include "TorchMLModelImplementation.hpp"
#include <cstddef>
#include <iostream>
#include "KIM_LogMacros.hpp"
#include <fstream>
#include <map>
#include <vector>

#define MAX_FILE_NUM 2
typedef double VecOfSize3[3];


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
    std::string const * paramFileName;
    std::string const * paramDirectory;

    // Model = y = mx +c

    std::cout<<"Param files count: " << numberParameterFiles <<"\n";
    if (numberParameterFiles > MAX_FILE_NUM){
        *ier = true;
        LOG_ERROR("Too many parameter files");
        return;
    }

    modelDriverCreate->GetParameterFileBasename(0,&paramFileName);
    modelDriverCreate->GetParameterFileDirectoryName(&paramDirectory);

    std::cout<<*paramFileName <<"\n";
    std::cout<<*paramDirectory <<"\n";

    auto filename = *paramDirectory + "/" + *paramFileName;
    std::ifstream filePtr(filename);
    filePtr >> m;filePtr >> c;
    filePtr.close();

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
    int numberOfSpecies = 1;
    std::string species = "Si";
    KIM::SpeciesName const specName1(species);

    //    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator> modelSpeciesMap;
    //    std::vector<KIM::SpeciesName> speciesNameVector;
    //
    //    speciesNameVector.push_back(species);
    //    // check for new species
    //    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>::const_iterator iIter = modelSpeciesMap.find(specName1);
    // all of the above is to remove species duplicates
    *ier = modelDriverCreate->SetSpeciesCode(specName1, 0);

    // Register Index settings-----------------------------------------------------------
    modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);
    std::cout << "indexing done\n";

    // Register Parameters --------------------------------------------------------------
    // Currently trying to create a simple Linear Model with two parameters from param file
    //Extent = number of entries in parameter pointer
    *ier = modelDriverCreate->SetParameterPointer(1,&m, "m", "slope");
    if (*ier) { LOG_ERROR("parameter m"); return;}
    *ier = modelDriverCreate->SetParameterPointer(1,&c, "c", "slope");
    if (*ier) { LOG_ERROR("parameter c"); return;}
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
    modelObject->m = 0.0;
    modelObject->c = 0.0;
    return false;
}

//******************************************************************************
// static member function
int TorchMLModel::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments){

    TorchMLModel * modelObject;
    modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

    auto m_ = modelObject->m;
    auto c_ = modelObject->c;

    int const * numberOfParticles;
    double * forces = NULL;
    double * coordinates = NULL;
    VecOfSize3 * coordinates3;
    double * energy;


//    int ier = 22;
    auto ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
            &numberOfParticles);
    ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::coordinates,
            &coordinates);
    ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
            &forces);
    ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
            &energy);

    coordinates3 = (VecOfSize3 *) coordinates;

    std::cout << *coordinates << "   " << *(coordinates + 2) << "   " << *(coordinates + 5) << "\n";
    std::cout << coordinates3[0][0] << "   " << coordinates3[0][2] << "   " << coordinates3[1][2] << "\n";
    *energy = m_ + c_;
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