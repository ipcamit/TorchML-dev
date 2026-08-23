#include "TorchMLModelDriver.hpp"
#include "TorchMLModelDriverImplementation.hpp"
#include "KIM_LogMacros.hpp"

#include <exception>
#include <string>

//==============================================================================
//
// This is the standard interface to KIM Model Drivers
//
//==============================================================================

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

extern "C" {
int model_driver_create(KIM::ModelDriverCreate * const modelDriverCreate,
                        KIM::LengthUnit const requestedLengthUnit,
                        KIM::EnergyUnit const requestedEnergyUnit,
                        KIM::ChargeUnit const requestedChargeUnit,
                        KIM::TemperatureUnit const requestedTemperatureUnit,
                        KIM::TimeUnit const requestedTimeUnit)
{
  // safer consistent idiomatic memory handling
  try
  {
    int ier;
    // read input files, convert units if needed, compute
    // interpolation coefficients, set cutoff, and publish parameters
    auto modelObject = std::make_unique<TorchMLModelDriver>(
        modelDriverCreate,
        requestedLengthUnit,
        requestedEnergyUnit,
        requestedChargeUnit,
        requestedTemperatureUnit,
        requestedTimeUnit,
        &ier);

    if (ier)
    {
      // constructor already reported the error
      return ier;
    }

    // register pointer to TorchMLModelDriverImplementation object in KIM object
    modelDriverCreate->SetModelBufferPointer(
        static_cast<void *>(modelObject.get()));
    modelObject.release();

    return false;
  }
  catch (const std::exception & e)
  {
    LOG_ERROR(std::string("TorchML model creation error: ") + e.what());
    return true;
  }
  catch (...)
  {
    LOG_ERROR("TorchML model creation failed with an unknown exception");
    return true;
  }
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
  implementation_ = std::make_unique<TorchMLModelDriverImplementation>(
      modelDriverCreate,
      requestedLengthUnit,
      requestedEnergyUnit,
      requestedChargeUnit,
      requestedTemperatureUnit,
      requestedTimeUnit,
      ier);
}

// **************************************************************************
TorchMLModelDriver::~TorchMLModelDriver() = default;

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

//******************************************************************************
// static member function
int TorchMLModelDriver::WriteParameterizedModel(
    const KIM::ModelWriteParameterizedModel * const
        modelWriteParameterizedModel)
{
  TorchMLModelDriver * modelObject;

  modelWriteParameterizedModel->GetModelBufferPointer(
      reinterpret_cast<void **>(&modelObject));

  return modelObject->implementation_->WriteParameterizedModel(
      modelWriteParameterizedModel);
}
