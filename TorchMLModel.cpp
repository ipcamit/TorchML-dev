#include "TorchMLModel.hpp"
#include "TorchMLModelImplementation.hpp"
#include <cstddef>


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
TorchMLModel::TorchMLModel(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
  implementation_ = new TorchMLModelImplementation(modelDriverCreate,
                                                      requestedLengthUnit,
                                                      requestedEnergyUnit,
                                                      requestedChargeUnit,
                                                      requestedTemperatureUnit,
                                                      requestedTimeUnit,
                                                      ier);
}

//******************************************************************************
TorchMLModel::~TorchMLModel() { delete implementation_; }

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
int TorchMLModel::Refresh(KIM::ModelRefresh * const modelRefresh)
{
  TorchMLModel * modelObject;
  modelRefresh->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

  return modelObject->implementation_->Refresh(modelRefresh);
}

//******************************************************************************
// static member function
int TorchMLModel::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  TorchMLModel * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

  return modelObject->implementation_->Compute(modelCompute,
                                               modelComputeArguments);
}

//******************************************************************************
// static member function
int TorchMLModel::ComputeArgumentsCreate(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
{
  TorchMLModel * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

  return modelObject->implementation_->ComputeArgumentsCreate(
      modelComputeArgumentsCreate);
}

//******************************************************************************
// static member function
int TorchMLModel::ComputeArgumentsDestroy(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy)
{
  TorchMLModel * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

  return modelObject->implementation_->ComputeArgumentsDestroy(
      modelComputeArgumentsDestroy);
}