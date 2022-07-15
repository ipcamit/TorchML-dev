//
// Created by amit on 7/12/22.
//

#ifndef TORCH_ML_MODEL_TORCHMLMODELIMPLEMENTATION_HPP
#define TORCH_ML_MODEL_TORCHMLMODELIMPLEMENTATION_HPP
#include "KIM_LogMacros.hpp"
#include "KIM_ModelDriverHeaders.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

#define DIMENSION 3
#define ONE 1.0
#define HALF 0.5

#define MAX_PARAMETER_FILES 1

#define PARAM_SHIFT_INDEX 0
#define PARAM_CUTOFFS_INDEX 1
#define PARAM_EPSILONS_INDEX 2
#define PARAM_SIGMAS_INDEX 3


//==============================================================================
//
// Type definitions, enumerations, and helper function prototypes
//
//==============================================================================

// type declaration for get neighbor functions
typedef int(GetNeighborFunction)(void const * const,
                                 int const,
                                 int * const,
                                 int const ** const);
// type declaration for vector of constant dimension
typedef double VectorOfSizeDIM[DIMENSION];
typedef double VectorOfSizeSix[6];

// helper routine declarations
void AllocateAndInitialize2DArray(double **& arrayPtr,
                                  int const extentZero,
                                  int const extentOne);
void Deallocate2DArray(double **& arrayPtr);

//==============================================================================
//
// Declaration ofTorchMLModelImplementation class
//
//==============================================================================

//******************************************************************************
class TorchMLModelImplementation
{
 public:
 TorchMLModelImplementation(
      KIM::ModelDriverCreate * const modelDriverCreate,
      KIM::LengthUnit const requestedLengthUnit,
      KIM::EnergyUnit const requestedEnergyUnit,
      KIM::ChargeUnit const requestedChargeUnit,
      KIM::TemperatureUnit const requestedTemperatureUnit,
      KIM::TimeUnit const requestedTimeUnit,
      int * const ier);
  ~TorchMLModelImplementation();  // no explicit Destroy() needed here

//  int Refresh(KIM::ModelRefresh * const modelRefresh);
  int Compute(KIM::ModelCompute const * const modelCompute,
              KIM::ModelComputeArguments const * const modelComputeArguments);
  int ComputeArgumentsCreate(KIM::ModelComputeArgumentsCreate * const
                                 modelComputeArgumentsCreate) const;
  int ComputeArgumentsDestroy(KIM::ModelComputeArgumentsDestroy * const
                                  modelComputeArgumentsDestroy) const;


 private:
  // Constant values that never change
  //   Set in constructor (via SetConstantValues)
  //
  //
  //TorchMLModelImplementation: constants
  int numberModelSpecies_;
  std::vector<int> modelSpeciesCodeList_;
  int numberUniqueSpeciesPairs_;


  // Constant values that are read from the input files and never change
  //   Set in constructor (via functions listed below)
  //
  //
  // Private Model Parameters
  //   Memory allocated in AllocatePrivateParameterMemory() (from constructor)
  //   Memory deallocated in destructor
  //   Data set in ReadParameterFile routines
  // none
  //
  // KIM API: Model Parameters whose (pointer) values never change
  //   Memory allocated in AllocateParameterMemory() (from constructor)
  //   Memory deallocated in destructor
  //   Data set in ReadParameterFile routines OR by KIM Simulator
  int shift_;
  double * cutoffs_;
  double * epsilons_;
  double * sigmas_;

  // Mutable values that only change when Refresh() executes
  //   Set in Refresh (via SetRefreshMutableValues)
  //
  //
  // KIM API: Model Parameters (can be changed directly by KIM Simulator)
  // none
  //
  //TorchMLModelImplementation: values (changed only by Refresh())
//  double influenceDistance_;
//  double ** cutoffsSq2D_;
//  int modelWillNotRequestNeighborsOfNoncontributingParticles_;
//  double ** fourEpsilonSigma6_2D_;
//  double ** fourEpsilonSigma12_2D_;
//  double ** twentyFourEpsilonSigma6_2D_;
//  double ** fortyEightEpsilonSigma12_2D_;
//  double ** oneSixtyEightEpsilonSigma6_2D_;
//  double ** sixTwentyFourEpsilonSigma12_2D_;
//  double ** shifts2D_;


  // Mutable values that can change with each call to Refresh() and Compute()
  //   Memory may be reallocated on each call
  //
  //
  //TorchMLModelImplementation: values that change
  int cachedNumberOfParticles_;


  // Helper methods
  //
  //
  // Related to constructor
  void AllocatePrivateParameterMemory();
  void AllocateParameterMemory();
  static int
  OpenParameterFiles(KIM::ModelDriverCreate * const modelDriverCreate,
                     int const numberParameterFiles,
                     FILE * parameterFilePointers[MAX_PARAMETER_FILES]);
  int ProcessParameterFiles(
      KIM::ModelDriverCreate * const modelDriverCreate,
      int const numberParameterFiles,
      FILE * const parameterFilePointers[MAX_PARAMETER_FILES]);
  void getNextDataLine(FILE * const filePtr,
                       char * const nextLine,
                       int const maxSize,
                       int * endOfFileFlag);
  static void
  CloseParameterFiles(int const numberParameterFiles,
                      FILE * const parameterFilePointers[MAX_PARAMETER_FILES]);
  int ConvertUnits(KIM::ModelDriverCreate * const modelDriverCreate,
                   KIM::LengthUnit const requestedLengthUnit,
                   KIM::EnergyUnit const requestedEnergyUnit,
                   KIM::ChargeUnit const requestedChargeUnit,
                   KIM::TemperatureUnit const requestedTemperatureUnit,
                   KIM::TimeUnit const requestedTimeUnit);
  int RegisterKIMModelSettings(
      KIM::ModelDriverCreate * const modelDriverCreate) const;
  int RegisterKIMComputeArgumentsSettings(
      KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
      const;
  int RegisterKIMParameters(KIM::ModelDriverCreate * const modelDriverCreate);
  int RegisterKIMFunctions(
      KIM::ModelDriverCreate * const modelDriverCreate) const;
  //
  // Related to Refresh()
//  template<class ModelObj>
//  int SetRefreshMutableValues(ModelObj * const modelObj);

  //
  // Related to Compute()
  int CheckParticleSpeciesCodes(KIM::ModelCompute const * const modelCompute,
                                int const * const particleSpeciesCodes) const;
  // compute functions
//  int Compute(KIM::ModelCompute const * const modelCompute,
//              KIM::ModelComputeArguments const * const modelComputeArguments,
//              const int * const particleSpeciesCodes,
//              const int * const particleContributing,
//              const VectorOfSizeDIM * const coordinates,
//              double * const energy,
//              VectorOfSizeDIM * const forces,
//              double * const particleEnergy,
//              VectorOfSizeSix virial,
//              VectorOfSizeSix * const particleVirial) const;
//};

//#undef KIM_LOGGER_OBJECT_NAME
//#define KIM_LOGGER_OBJECT_NAME modelCompute
////
//int TorchMLModelImplementation::Compute(
//    KIM::ModelCompute const * const modelCompute,
//    KIM::ModelComputeArguments const * const modelComputeArguments,
//    const int * const particleSpeciesCodes,
//    const int * const particleContributing,
//    const VectorOfSizeDIM * const coordinates,
//    double * const energy,
//    VectorOfSizeDIM * const forces,
//    double * const particleEnergy,
//    VectorOfSizeSix virial,
//    VectorOfSizeSix * const particleVirial) const
//{
//  int ier = false;
//}

#endif //TORCH_ML_MODEL_TORCHMLMODELIMPLEMENTATION_HPP
