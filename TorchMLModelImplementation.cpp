#include <cmath>
#include <cstring>
#include <iostream>  // IWYU pragma: keep  // BUG WORK-AROUND
#include <map>
#include <sstream>
#include <string>

#include "KIM_ModelDriverHeaders.hpp"
#include "TorchMLModelDriver.hpp"
#include "TorchMLModelImplementation.hpp"

#define MAXLINE 1024
#define IGNORE_RESULT(fn) \
  if (fn) {}


//==============================================================================
//
// Implementation of TorchMLModelImplementation public member functions
//
//==============================================================================

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
//
TorchMLModelImplementation::TorchMLModelImplementation(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier) :
    cachedNumberOfParticles_(0)
{
  FILE * parameterFilePointers[MAX_PARAMETER_FILES];
  int numberParameterFiles;
  modelDriverCreate->GetNumberOfParameterFiles(&numberParameterFiles);
  *ier = OpenParameterFiles(
      modelDriverCreate, numberParameterFiles, parameterFilePointers);
  if (*ier) return;

  *ier = ProcessParameterFiles(
      modelDriverCreate, numberParameterFiles, parameterFilePointers);
  CloseParameterFiles(numberParameterFiles, parameterFilePointers);
  if (*ier) return;

  *ier = ConvertUnits(modelDriverCreate,
                      requestedLengthUnit,
                      requestedEnergyUnit,
                      requestedChargeUnit,
                      requestedTemperatureUnit,
                      requestedTimeUnit);
  if (*ier) return;

  *ier = RegisterKIMModelSettings(modelDriverCreate);
  if (*ier) return;

  *ier = RegisterKIMParameters(modelDriverCreate);
  if (*ier) return;

  *ier = RegisterKIMFunctions(modelDriverCreate);
  if (*ier) return;

  // everything is good
  *ier = false;
  return;
}

//******************************************************************************
//#undef KIM_LOGGER_OBJECT_NAME
//#define KIM_LOGGER_OBJECT_NAME modelRefresh
////
//int TorchMLModelImplementation::Refresh(
//    KIM::ModelRefresh * const modelRefresh)
//{
//  int ier;
//
//  ier = SetRefreshMutableValues(modelRefresh);
//  if (ier) return ier;
//
//  // nothing else to do for this case
//
//  // everything is good
//  ier = false;
//  return ier;
//}

//******************************************************************************
int TorchMLModelImplementation::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
//  int ier;
//
//  // KIM API Model Input compute flags
//  bool isComputeProcess_dEdr = false;
//  bool isComputeProcess_d2Edr2 = false;
//  //
//  // KIM API Model Output compute flags
//  bool isComputeEnergy = false;
//  bool isComputeForces = false;
//  bool isComputeParticleEnergy = false;
//  bool isComputeVirial = false;
//  bool isComputeParticleVirial = false;
//  //
//  // KIM API Model Input
//  int const * particleSpeciesCodes = NULL;
//  int const * particleContributing = NULL;
//  VectorOfSizeDIM const * coordinates = NULL;
//  //
//  // KIM API Model Output
//  double * energy = NULL;
//  double * particleEnergy = NULL;
//  VectorOfSizeDIM * forces = NULL;
//  VectorOfSizeSix * virial = NULL;
//  VectorOfSizeSix * particleVirial = NULL;
//  ier = SetComputeMutableValues(modelComputeArguments,
//                                isComputeProcess_dEdr,
//                                isComputeProcess_d2Edr2,
//                                isComputeEnergy,
//                                isComputeForces,
//                                isComputeParticleEnergy,
//                                isComputeVirial,
//                                isComputeParticleVirial,
//                                particleSpeciesCodes,
//                                particleContributing,
//                                coordinates,
//                                energy,
//                                particleEnergy,
//                                forces,
//                                virial,
//                                particleVirial);
//  if (ier) return ier;

  // Skip this check for efficiency
  //
  // ier = CheckParticleSpecies(modelComputeArguments, particleSpeciesCodes);
  // if (ier) return ier;

//  bool const isShift = (1 == shift_);
//
////#include "TorchMLModelImplementationComputeDispatch.cpp"
//  return ier;
}

//******************************************************************************
int TorchMLModelImplementation::ComputeArgumentsCreate(
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate) const
{
  int ier;

  ier = RegisterKIMComputeArgumentsSettings(modelComputeArgumentsCreate);
  if (ier) return ier;

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int TorchMLModelImplementation::ComputeArgumentsDestroy(
    KIM::ModelComputeArgumentsDestroy * const
    /* modelComputeArgumentsDestroy */) const
{
  int ier;

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}

//==============================================================================
//
// Implementation of TorchMLModelImplementation private member functions
//
//==============================================================================

//******************************************************************************
void TorchMLModelImplementation::AllocateParameterMemory()
{  // allocate memory for data
//  cutoffs_ = new double[numberUniqueSpeciesPairs_];
//  AllocateAndInitialize2DArray(
//      cutoffsSq2D_, numberModelSpecies_, numberModelSpecies_);
//
//  epsilons_ = new double[numberUniqueSpeciesPairs_];
//  sigmas_ = new double[numberUniqueSpeciesPairs_];
//  AllocateAndInitialize2DArray(
//      fourEpsilonSigma6_2D_, numberModelSpecies_, numberModelSpecies_);
//  AllocateAndInitialize2DArray(
//      fourEpsilonSigma12_2D_, numberModelSpecies_, numberModelSpecies_);
//  AllocateAndInitialize2DArray(
//      twentyFourEpsilonSigma6_2D_, numberModelSpecies_, numberModelSpecies_);
//  AllocateAndInitialize2DArray(
//      fortyEightEpsilonSigma12_2D_, numberModelSpecies_, numberModelSpecies_);
//  AllocateAndInitialize2DArray(
//      oneSixtyEightEpsilonSigma6_2D_, numberModelSpecies_, numberModelSpecies_);
//  AllocateAndInitialize2DArray(sixTwentyFourEpsilonSigma12_2D_,
//                               numberModelSpecies_,
//                               numberModelSpecies_);
//
//  AllocateAndInitialize2DArray(
//      shifts2D_, numberModelSpecies_, numberModelSpecies_);
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
//
int TorchMLModelImplementation::OpenParameterFiles(
    KIM::ModelDriverCreate * const modelDriverCreate,
    int const numberParameterFiles,
    FILE * parameterFilePointers[MAX_PARAMETER_FILES])
{
  int ier;

  if (numberParameterFiles > MAX_PARAMETER_FILES)
  {
    ier = true;
    LOG_ERROR("TorchMLModel given too many parameter files");
    return ier;
  }

  std::string const * paramFileDirName;
  modelDriverCreate->GetParameterFileDirectoryName(&paramFileDirName);
  for (int i = 0; i < numberParameterFiles; ++i)
  {
    std::string const * paramFileName;
    ier = modelDriverCreate->GetParameterFileBasename(i, &paramFileName);
    if (ier)
    {
      LOG_ERROR("Unable to get parameter file name");
      return ier;
    }
    std::string filename = *paramFileDirName + "/" + *paramFileName;
    parameterFilePointers[i] = fopen(filename.c_str(), "r");
    if (parameterFilePointers[i] == 0)
    {
      char message[MAXLINE];
      sprintf(message,
              "TorchMLModel parameter file number %d cannot be opened",
              i);
      ier = true;
      LOG_ERROR(message);
      for (int j = i - 1; j >= 0; --j) { fclose(parameterFilePointers[j]); }
      return ier;
    }
  }

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
//
int TorchMLModelImplementation::ProcessParameterFiles(
    KIM::ModelDriverCreate * const modelDriverCreate,
    int const /* numberParameterFiles */,
    FILE * const parameterFilePointers[MAX_PARAMETER_FILES])
{
//  getNextDataLine(
//      parameterFilePointers[0], nextLinePtr, MAXLINE, &endOfFileFlag);
//  ier = sscanf(nextLine, "%d %d", &N, &shift_);
//  if (ier != 2)
//  {
//    sprintf(nextLine, "unable to read first line of the parameter file");
//    ier = true;
//    LOG_ERROR(nextLine);
//    fclose(parameterFilePointers[0]);
//    return ier;
//  }
//  numberModelSpecies_ = N;
//  numberUniqueSpeciesPairs_
//      = ((numberModelSpecies_ + 1) * numberModelSpecies_) / 2;
//  AllocateParameterMemory();

  // set all values in the arrays to -1 for mixing later
  // keep track of known species
//  std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>
//      modelSpeciesMap;
//  std::vector<KIM::SpeciesName> speciesNameVector;
//    // convert species strings to proper type instances
//    KIM::SpeciesName const specName1(spec1);
//    KIM::SpeciesName const specName2(spec2);

    // check for new species
//    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>::
//        const_iterator iIter
//        = modelSpeciesMap.find(specName1);
//      ier = modelDriverCreate->SetSpeciesCode(specName1, index);
}

//******************************************************************************
void TorchMLModelImplementation::getNextDataLine(FILE * const filePtr,
                                                    char * nextLinePtr,
                                                    int const maxSize,
                                                    int * endOfFileFlag)
{
  do {
    if (fgets(nextLinePtr, maxSize, filePtr) == NULL)
    {
      *endOfFileFlag = 1;
      break;
    }
    while ((nextLinePtr[0] == ' ' || nextLinePtr[0] == '\t')
           || (nextLinePtr[0] == '\n' || nextLinePtr[0] == '\r'))
    {
      nextLinePtr = (nextLinePtr + 1);
    }
  } while ((strncmp("#", nextLinePtr, 1) == 0) || (strlen(nextLinePtr) == 0));
}

//******************************************************************************
void TorchMLModelImplementation::CloseParameterFiles(
    int const numberParameterFiles,
    FILE * const parameterFilePointers[MAX_PARAMETER_FILES])
{
  for (int i = 0; i < numberParameterFiles; ++i)
    fclose(parameterFilePointers[i]);
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
//
int TorchMLModelImplementation::ConvertUnits(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit)
{
  int ier;

  // define default base units
  KIM::LengthUnit fromLength = KIM::LENGTH_UNIT::A;
  KIM::EnergyUnit fromEnergy = KIM::ENERGY_UNIT::eV;
  KIM::ChargeUnit fromCharge = KIM::CHARGE_UNIT::e;
  KIM::TemperatureUnit fromTemperature = KIM::TEMPERATURE_UNIT::K;
  KIM::TimeUnit fromTime = KIM::TIME_UNIT::ps;

  // changing units of cutoffs and sigmas
  double convertLength = 1.0;
  ier = KIM::ModelDriverCreate::ConvertUnit(fromLength,
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
  if (ier)
  {
    LOG_ERROR("Unable to convert length unit");
    return ier;
  }
  if (convertLength != ONE)
  {
    for (int i = 0; i < numberUniqueSpeciesPairs_; ++i)
    {
      cutoffs_[i] *= convertLength;  // convert to active units
      sigmas_[i] *= convertLength;  // convert to active units
    }
  }
  // changing units of epsilons
  double convertEnergy = 1.0;
  ier = KIM::ModelDriverCreate::ConvertUnit(fromLength,
                                            fromEnergy,
                                            fromCharge,
                                            fromTemperature,
                                            fromTime,
                                            requestedLengthUnit,
                                            requestedEnergyUnit,
                                            requestedChargeUnit,
                                            requestedTemperatureUnit,
                                            requestedTimeUnit,
                                            0.0,
                                            1.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            &convertEnergy);
  if (ier)
  {
    LOG_ERROR("Unable to convert energy unit");
    return ier;
  }
  if (convertEnergy != ONE)
  {
    for (int i = 0; i < numberUniqueSpeciesPairs_; ++i)
    {
      epsilons_[i] *= convertEnergy;  // convert to active units
    }
  }

  // register units
  ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                    requestedEnergyUnit,
                                    KIM::CHARGE_UNIT::unused,
                                    KIM::TEMPERATURE_UNIT::unused,
                                    KIM::TIME_UNIT::unused);
  if (ier)
  {
    LOG_ERROR("Unable to set units to requested values");
    return ier;
  }

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int TorchMLModelImplementation::RegisterKIMModelSettings(
    KIM::ModelDriverCreate * const modelDriverCreate) const
{
  // register numbering
  int error = modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);

  return error;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate
//
int TorchMLModelImplementation::RegisterKIMComputeArgumentsSettings(
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate) const
{
  // register arguments
  LOG_INFORMATION("Register argument supportStatus");
  int error = modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialForces,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialParticleVirial,
                  KIM::SUPPORT_STATUS::optional);


  // register callbacks
  LOG_INFORMATION("Register callback supportStatus");
  error = error
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
              KIM::SUPPORT_STATUS::optional)
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
              KIM::SUPPORT_STATUS::optional);

  return error;
}

#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
//
int TorchMLModelImplementation::RegisterKIMParameters(
    KIM::ModelDriverCreate * const modelDriverCreate)
{
  int ier = false;

  // publish parameters (order is important)
  ier = modelDriverCreate->SetParameterPointer(
      1,
      &shift_,
      "shift",
      "If (shift == 1), all LJ potentials are shifted to zero energy "
      "at their respective cutoff distance.  Otherwise, no shifting is "
      "performed.");
  if (ier)
  {
    LOG_ERROR("set_parameter shift");
    return ier;
  }

  ier = modelDriverCreate->SetParameterPointer(
      numberUniqueSpeciesPairs_,
      cutoffs_,
      "cutoffs",
      "Lower-triangular matrix (of size N= ");
  if (ier)
  {
    LOG_ERROR("set_parameter cutoffs");
    return ier;
  }
  ier = modelDriverCreate->SetParameterPointer(
      numberUniqueSpeciesPairs_,
      epsilons_,
      "epsilons",
      "Lower-triangular matrix (of size N= ");
  if (ier)
  {
    LOG_ERROR("set_parameter epsilons");
    return ier;
  }
  ier = modelDriverCreate->SetParameterPointer(
      numberUniqueSpeciesPairs_,
      sigmas_,
      "sigmas",
      "Lower-triangular matrix (of size N= ");
  if (ier)
  {
    LOG_ERROR("set_parameter sigmas");
    return ier;
  }

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int TorchMLModelImplementation::RegisterKIMFunctions(
    KIM::ModelDriverCreate * const modelDriverCreate) const
{
  int error;

  // Use function pointer definitions to verify correct prototypes
  KIM::ModelDestroyFunction * destroy = TorchMLModel::Destroy;
//  KIM::ModelRefreshFunction * refresh = TorchMLModel::Refresh;
  KIM::ModelComputeFunction * compute = TorchMLModel::Compute;
  KIM::ModelComputeArgumentsCreateFunction * CACreate
      = TorchMLModel::ComputeArgumentsCreate;
  KIM::ModelComputeArgumentsDestroyFunction * CADestroy
      = TorchMLModel::ComputeArgumentsDestroy;

  // register the destroy() and reinit() functions
  error = modelDriverCreate->SetRoutinePointer(
              KIM::MODEL_ROUTINE_NAME::Destroy,
              KIM::LANGUAGE_NAME::cpp,
              true,
              reinterpret_cast<KIM::Function *>(destroy))
//          || modelDriverCreate->SetRoutinePointer(
//              KIM::MODEL_ROUTINE_NAME::Refresh,
//              KIM::LANGUAGE_NAME::cpp,
//              true,
//              reinterpret_cast<KIM::Function *>(refresh))
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
  return error;
}


//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments
//
//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCompute
int TorchMLModelImplementation::CheckParticleSpeciesCodes(
    KIM::ModelCompute const * const modelCompute,
    int const * const particleSpeciesCodes) const
{
  int ier;
  for (int i = 0; i < cachedNumberOfParticles_; ++i)
  {
    if ((particleSpeciesCodes[i] < 0)
        || (particleSpeciesCodes[i] >= numberModelSpecies_))
    {
      ier = true;
      LOG_ERROR("unsupported particle species codes detected");
      return ier;
    }
  }

  // everything is good
  ier = false;
  return ier;
}


//==============================================================================
//
// Implementation of helper functions
//
//==============================================================================

//******************************************************************************
void AllocateAndInitialize2DArray(double **& arrayPtr,
                                  int const extentZero,
                                  int const extentOne)
{  // allocate memory and set pointers
  arrayPtr = new double *[extentZero];
  arrayPtr[0] = new double[extentZero * extentOne];
  for (int i = 1; i < extentZero; ++i)
  {
    arrayPtr[i] = arrayPtr[i - 1] + extentOne;
  }

  // initialize
  for (int i = 0; i < extentZero; ++i)
  {
    for (int j = 0; j < extentOne; ++j) { arrayPtr[i][j] = 0.0; }
  }
}

//******************************************************************************
void Deallocate2DArray(double **& arrayPtr) {  // deallocate memory
    if (arrayPtr != NULL) delete[] arrayPtr[0];
    delete[] arrayPtr;

    // nullify pointer
    arrayPtr = NULL;
}