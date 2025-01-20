#include "TorchMLModelDriver.hpp"
#include "TorchMLModelDriverImplementation.hpp"

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
  auto modelObject = new TorchMLModelDriver(modelDriverCreate,
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

  // register pointer to TorchMLModelDriverImplementation object in KIM object
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

// ****************************** ********* **********************************
TorchMLModelDriver::TorchMLModelDriver(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
  implementation_
      = new TorchMLModelDriverImplementation(modelDriverCreate,
                                             requestedLengthUnit,
                                             requestedEnergyUnit,
                                             requestedChargeUnit,
                                             requestedTemperatureUnit,
                                             requestedTimeUnit,
                                             ier);
}

// **************************************************************************
TorchMLModelDriver::~TorchMLModelDriver() { delete implementation_; }

//******************************************************************************
// static member function
int TorchMLModelDriver::Destroy(KIM::ModelDestroy * const modelDestroy)
{
  TorchMLModelDriver * modelObject;
  modelDestroy->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  delete modelObject;
  return false;
}

//******************************************************************************
// static member function
int TorchMLModelDriver::Refresh(KIM::ModelRefresh * const modelRefresh)
{
  TorchMLModelDriver * modelObject;
  modelRefresh->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

  return modelObject->implementation_->Refresh(modelRefresh);
}

//******************************************************************************
// static member function
int TorchMLModelDriver::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  TorchMLModelDriver * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  return modelObject->implementation_->Compute(modelComputeArguments);
}

//******************************************************************************
// static member function
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate

int TorchMLModelDriver::ComputeArgumentsCreate(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
{
  TorchMLModelDriver * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  return modelObject->implementation_->ComputeArgumentsCreate(
      modelComputeArgumentsCreate);
}

//******************************************************************************
// static member function
int TorchMLModelDriver::ComputeArgumentsDestroy(
    KIM::ModelCompute const * modelCompute,
    KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy)
{
  TorchMLModelDriver * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  return modelObject->implementation_->ComputeArgumentsDestroy(
      modelComputeArgumentsDestroy);
}