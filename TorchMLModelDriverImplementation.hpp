//
// Created by amit on 7/12/22.
//

#ifndef TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP
#define TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP

#include "KIM_ModelDriverHeaders.hpp"
#include "MLModel.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>


#ifdef USE_LIBDESC
#include "Descriptors.hpp"

using namespace Descriptor;
#endif

enum class PreprocessingMode { None, Descriptor, Graph };

class TorchMLModelDriverImplementation
{
 public:
  // All file params are public
  double influence_distance, cutoff_distance;
  int n_elements, n_layers;
  std::vector<std::string> elements_list;
  PreprocessingMode preprocessing = PreprocessingMode::None;
  std::string model_name;
  bool returns_forces;
  std::string descriptor_name = "None";
  std::string descriptor_param_file = "None";
  std::string descriptor_param_file_content;
  std::string fully_qualified_model_name;

  TorchMLModelDriverImplementation(
      KIM::ModelDriverCreate * modelDriverCreate,
      KIM::LengthUnit requestedLengthUnit,
      KIM::EnergyUnit requestedEnergyUnit,
      KIM::ChargeUnit requestedChargeUnit,
      KIM::TemperatureUnit requestedTemperatureUnit,
      KIM::TimeUnit requestedTimeUnit,
      int * ier);

  ~TorchMLModelDriverImplementation();

  int Refresh(KIM::ModelRefresh * modelRefresh);
  int Refresh(KIM::ModelDriverCreate * modelRefresh);

  int Compute(KIM::ModelComputeArguments const * modelComputeArguments);

  int ComputeArgumentsCreate(
      KIM::ModelComputeArgumentsCreate * modelComputeArgumentsCreate);

  int ComputeArgumentsDestroy(
      KIM::ModelComputeArgumentsDestroy * modelComputeArgumentsDestroy);

  int WriteParameterizedModel(KIM::ModelWriteParameterizedModel const * const
                                  modelWriteParameterizedModel) const;

 private:
  // Derived or assigned variables are private
  int modelWillNotRequestNeighborsOfNoncontributingParticles_;
  int n_contributing_atoms;
  int number_of_inputs;

  bool map_species_to_z;

  std::vector<std::int64_t> species_atomic_number;
  std::vector<std::int64_t> contraction_array;

  std::unique_ptr<MLModel> ml_model;
  KIM::EnergyUnit model_energy_unit_ = KIM::ENERGY_UNIT::eV;
  KIM::LengthUnit model_length_unit_ = KIM::LENGTH_UNIT::A;
  double energy_factor = 1.0;
  double force_factor = 1.0;
  double length_factor = 1.0;
  std::shared_ptr<double[]> positions_buffer;
  std::shared_ptr<double[]> force_buffer;
  bool lengthUnitConversionRequested; //default eV and A
  bool energyUnitConversionRequested; //default eV and A

#ifdef USE_LIBDESC
  AvailableDescriptor descriptor_kind;
  std::unique_ptr<DescriptorKind> descriptor;
#endif
  std::vector<int> num_neighbors_;
  std::vector<int> neighbor_list;
  std::vector<int> z_map;

  std::vector<double> descriptor_array;
  std::vector<std::vector<std::int64_t> > graph_edge_indices;

  int
  updateNeighborList(KIM::ModelComputeArguments const * modelComputeArguments);

  int
  setDefaultInputs(const KIM::ModelComputeArguments * modelComputeArguments);

  int
  setDescriptorInputs(const KIM::ModelComputeArguments * modelComputeArguments);

  int setGraphInputs(const KIM::ModelComputeArguments * modelComputeArguments);

#if 0  // Held for later reintroduction after unit conversion support.
  int setGraphInputsFast(const KIM::ModelComputeArguments * modelComputeArguments);
#endif

  void readParametersFile(KIM::ModelDriverCreate * modelDriverCreate,
                          int * ier);


  void unitConversion(KIM::ModelDriverCreate * modelDriverCreate,
                             KIM::LengthUnit requestedLengthUnit,
                             KIM::EnergyUnit requestedEnergyUnit,
                             KIM::ChargeUnit requestedChargeUnit,
                             KIM::TemperatureUnit requestedTemperatureUnit,
                             KIM::TimeUnit requestedTimeUnit,
                             int * ier);

  void setSpecies(KIM::ModelDriverCreate * modelDriverCreate, int * ier);

  static void
  registerFunctionPointers(KIM::ModelDriverCreate * modelDriverCreate,
                           int * ier);

  int
  preprocessInputs(KIM::ModelComputeArguments const * modelComputeArguments);

  int postprocessOutputs(KIM::ModelComputeArguments const *);

  int Run(KIM::ModelComputeArguments const * modelComputeArguments);

  int contributingAtomCounts(
      KIM::ModelComputeArguments const * modelComputeArguments);
};

int sym_to_z(std::string &);

// For hashing unordered_set of pairs
// https://arxiv.org/pdf/2105.10752.pdf
// It seems like this might not be the best approach for hashing edges
// But surprisingly it is.

// === Benchmarking (micro s) ===
// N_edgs	 Mrgsrt(arr)	Cantor*	 BoostHash	32bitPackHash	Cantor,bidirectional
// 10^1	   13	          7	       1	        2	            7
// 10^2	   14	          22       25         30            31
// 10^3    272          158      177        195           315
// 10^4    13222        1308     1511       1595          2498
// 10^5    1347793      25009    30921      28328         46013
// 10^6    142392300    422934   489779     466556        548133
// * current method
// Most likely FMAs kinda instructions make CantorPairs on par with bitwise
// hashes
//
class CantorPairing
{
 public:
  int64_t operator()(const std::array<long, 2> & t) const
  {
    int64_t k1 = t[0];
    int64_t k2 = t[1];
    // int64_t kmin = std::min(k1, k2);
    // int64_t ksum = k1 + k2 + 1;
    //
    // return ((ksum * ksum - ksum % 2) + kmin) / 4;
    int64_t sum = k1 + k2;
    int64_t triangleNumber = sum * (sum + 1) / 2;
    return triangleNumber + k2;
  }
};


#endif  // TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP
