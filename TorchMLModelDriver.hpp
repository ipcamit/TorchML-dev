//
// Created by amit on 7/12/22.
// TODO PIMPL implementation
// Scheme: snake_case for implemented variables
// Methods : camelCase
//

#ifndef TORCH_ML_MODEL_H
#define TORCH_ML_MODEL_H

#include "KIM_ModelDriverHeaders.hpp"
#include "MLModel.hpp"
#include "descriptors.hpp"
//TODO Remove the SYMFUN calls below. only for temp workaround
//#include "SymFun/SymFun.hpp"
#include <torch/torch.h>

extern "C" {
int model_driver_create(KIM::ModelDriverCreate *modelDriverCreate,
                        KIM::LengthUnit requestedLengthUnit,
                        KIM::EnergyUnit requestedEnergyUnit,
                        KIM::ChargeUnit requestedChargeUnit,
                        KIM::TemperatureUnit requestedTemperatureUnit,
                        KIM::TimeUnit requestedTimeUnit);
}

//class TorchMLModelImplementation;

class TorchMLModelDriver {
public:
    // All file params are public
    double influence_distance, cutoff_distance;
    int n_elements, n_layers;
    std::vector<std::string> elements_list;
    std::string preprocessing;
    std::string model_name;
    bool returns_forces;
    std::string descriptor_name = "None";
    std::string descriptor_param_file = "None";

    TorchMLModelDriver(KIM::ModelDriverCreate *modelDriverCreate,
                       KIM::LengthUnit requestedLengthUnit,
                       KIM::EnergyUnit requestedEnergyUnit,
                       KIM::ChargeUnit requestedChargeUnit,
                       KIM::TemperatureUnit requestedTemperatureUnit,
                       KIM::TimeUnit requestedTimeUnit,
                       int *ier);
  ~TorchMLModelDriver();

    // no need to make these "extern" since KIM will only access them
    // via function pointers.  "static" is required so that there is not
    // an implicit this pointer added to the prototype by the C++ compiler
    static int Destroy(KIM::ModelDestroy *modelDestroy);

    static int Refresh(KIM::ModelRefresh *modelRefresh);

    static int Compute(KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArguments const *modelComputeArguments);

    static int ComputeArgumentsCreate(
            KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArgumentsCreate *modelComputeArgumentsCreate);

    static int ComputeArgumentsDestroy(
            KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArgumentsDestroy *modelComputeArgumentsDestroy);

private:
    // Derived or assigned variables are private
    int modelWillNotRequestNeighborsOfNoncontributingParticles_;
    int n_contributing_atoms;
    int number_of_inputs;
    int * species_atomic_number;
    int64_t * contraction_array;

    MLModel *mlModel;

    Descriptor *descriptor;

    std::vector<int> num_neighbors_;
    std::vector<int> neighbor_list;
    std::vector<int> z_map;

    double * descriptor_array;
    long ** graph_edge_indices;

    void updateNeighborList(KIM::ModelComputeArguments const *modelComputeArguments, int numberOfParticles);

    void setDefaultInputs(const KIM::ModelComputeArguments * modelComputeArguments);
    void setDescriptorInputs(const KIM::ModelComputeArguments * modelComputeArguments);
    void setGraphInputs(const KIM::ModelComputeArguments * modelComputeArguments);

    void readParametersFile(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    static void unitConversion(KIM::ModelDriverCreate *modelDriverCreate,
                               KIM::LengthUnit requestedLengthUnit,
                               KIM::EnergyUnit requestedEnergyUnit,
                               KIM::ChargeUnit requestedChargeUnit,
                               KIM::TemperatureUnit requestedTemperatureUnit,
                               KIM::TimeUnit requestedTimeUnit,
                               int *ier);

    void setSpecies(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    static void registerFunctionPointers(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    void preprocessInputs(KIM::ModelComputeArguments const *modelComputeArguments);
    void postprocessOutputs(c10::IValue&, KIM::ModelComputeArguments const *);

    void Run(KIM::ModelComputeArguments const *modelComputeArguments);
    void contributingAtomCounts(KIM::ModelComputeArguments const *modelComputeArguments);
    // This is only supplementary function to be called by other function, so
    // this is following different naming convention.
    void graph_set_to_graph_array(std::vector<std::set<std::tuple<long, long> > > &);
    // TorchMLModelImplementation * implementation_;
};

int sym_to_z(std::string & sym){
    // TODO more idiomatic handling of species. Ask Ryan
    if (sym == "H")   return 1;
    if (sym == "He")  return 2;
    if (sym == "Li")  return 3;
    if (sym == "Be")  return 4;
    if (sym == "B")   return 5;
    if (sym == "C")   return 6;
    if (sym == "N")   return 7;
    if (sym == "O")   return 8;
    if (sym == "F")   return 9;
    if (sym == "Ne")  return 10;
    if (sym == "Na")  return 11;
    if (sym == "Mg")  return 12;
    if (sym == "Al")  return 13;
    if (sym == "Si")  return 14;
    if (sym == "P")   return 15;
    if (sym == "S")   return 16;
    if (sym == "Cl")  return 17;
    if (sym == "A")   return 18;
    if (sym == "K")   return 19;
    if (sym == "Ca")  return 20;
    if (sym == "Sc")  return 21;
    if (sym == "Ti")  return 22;
    if (sym == "V")   return 23;
    if (sym == "Cr")  return 24;
    if (sym == "Mn")  return 25;
    if (sym == "Fe")  return 26;
    if (sym == "Co")  return 27;
    if (sym == "Ni")  return 28;
    if (sym == "Cu")  return 29;
    if (sym == "Zn")  return 30;
    if (sym == "Ga")  return 31;
    if (sym == "Ge")  return 32;
    if (sym == "As")  return 33;
    if (sym == "Se")  return 34;
    if (sym == "Br")  return 35;
    if (sym == "Kr")  return 36;
    if (sym == "Rb")  return 37;
    if (sym == "Sr")  return 38;
    if (sym == "Y")   return 39;
    if (sym == "Zr")  return 40;
    if (sym == "Nb")  return 41;
    if (sym == "Mo")  return 42;
    if (sym == "Tc")  return 43;
    if (sym == "Ru")  return 44;
    if (sym == "Rh")  return 45;
    if (sym == "Pd")  return 46;
    if (sym == "Ag")  return 47;
    if (sym == "Cd")  return 48;
    if (sym == "In")  return 49;
    if (sym == "Sn")  return 50;
    if (sym == "Sb")  return 51;
    if (sym == "Te")  return 52;
    if (sym == "I")   return 53;
    if (sym == "Xe")  return 54;
    if (sym == "Cs")  return 55;
    if (sym == "Ba")  return 56;
    if (sym == "La")  return 57;
    if (sym == "Ce")  return 58;
    if (sym == "Pr")  return 59;
    if (sym == "Nd")  return 60;
    if (sym == "Pm")  return 61;
    if (sym == "Sm")  return 62;
    if (sym == "Eu")  return 63;
    if (sym == "Gd")  return 64;
    if (sym == "Tb")  return 65;
    if (sym == "Dy")  return 66;
    if (sym == "Ho")  return 67;
    if (sym == "Er")  return 68;
    if (sym == "Tm")  return 69;
    if (sym == "Yb")  return 70;
    if (sym == "Lu")  return 71;
    if (sym == "Hf")  return 72;
    if (sym == "Ta")  return 73;
    if (sym == "W")   return 74;
    if (sym == "Re")  return 75;
    if (sym == "Os")  return 76;
    if (sym == "Ir")  return 77;
    if (sym == "Pt")  return 78;
    if (sym == "Au")  return 79;
    if (sym == "Hg")  return 80;
    if (sym == "Ti")  return 81;
    if (sym == "Pb")  return 82;
    if (sym == "Bi")  return 83;
    if (sym == "Po")  return 84;
    if (sym == "At")  return 85;
    if (sym == "Rn")  return 86;
    if (sym == "Fr")  return 87;
    if (sym == "Ra")  return 88;
    if (sym == "Ac")  return 89;
    if (sym == "Th")  return 90;
    if (sym == "Pa")  return 91;
    if (sym == "U")   return 92;
}
#endif //TORCH_ML_MODEL_H