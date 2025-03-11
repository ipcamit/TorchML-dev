#include "TorchMLModelDriverImplementation.hpp"
#include "KIM_LogMacros.hpp"
#include "TorchMLModelDriver.hpp"
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

#ifndef DISABLE_GRAPH
#include <torchscatter/scatter.h>
#endif

#define MAX_FILE_NUM 3

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate


TorchMLModelDriverImplementation::TorchMLModelDriverImplementation(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
  *ier = false;
  // initialize members to remove warning----
  influence_distance = 0.0;
  n_elements = 0;
  ml_model = nullptr;
  returns_forces = false;
  cutoff_distance = 0.0;
  n_layers = 0;
  n_contributing_atoms = 0;
  number_of_inputs = 0;
  auto exec_device = std::getenv("KIM_MODEL_EXECUTION_DEVICE");
  auto device = exec_device ? std::string {exec_device} : std::string {"cpu"};

  // Read parameter files from model driver
  // --------------------------------------- also initialize the ml_model
  readParametersFile(modelDriverCreate, ier);
  // Load Torch Model
  // ----------------------------------------------------------------
  ml_model = std::unique_ptr<MLModel>(
      MLModel::create(fully_qualified_model_name, device, number_of_inputs));
  LOG_INFORMATION("Loaded Torch model and set to eval");
  LOG_DEBUG("Read Param files");
  if (*ier) return;

  // Unit conversions
  // -----------------------------------------------------------------
  unitConversion(modelDriverCreate,
                 requestedLengthUnit,
                 requestedEnergyUnit,
                 requestedChargeUnit,
                 requestedTemperatureUnit,
                 requestedTimeUnit,
                 ier);
  LOG_DEBUG("Registered Unit Conversion");
  if (*ier) return;

  // Set Influence distance
  // ---------------------------------------------------------
  if (preprocessing == "graph")
  {
#ifndef DISABLE_GRAPH
    modelWillNotRequestNeighborsOfNoncontributingParticles_
        = static_cast<int>(false);
  }
  else
  {
    modelWillNotRequestNeighborsOfNoncontributingParticles_
        = static_cast<int>(true);
  }
#else
    LOG_ERROR("Graph preprocessing is not supported in this build");
    *ier = true;
    return;
  }
#endif
  modelDriverCreate->SetInfluenceDistancePointer(&influence_distance);
  modelDriverCreate->SetNeighborListPointers(
      1,
      &cutoff_distance,
      &modelWillNotRequestNeighborsOfNoncontributingParticles_);

  // Species code
  // --------------------------------------------------------------------
  setSpecies(modelDriverCreate, ier);
  LOG_DEBUG("Registered Species");
  if (*ier) return;

  // Register Index
  // settings-----------------------------------------------------------
  modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);

  // Register Parameters
  // -------------------------------------------------------------- All model
  // parameters are inside model So nothing to do here? Just Registering
  // n_elements lest KIM complaints. TODO Discuss with Ryan?
  *ier = modelDriverCreate->SetParameterPointer(
      1, &n_elements, "n_elements", "Number of elements");
  // Should there be void * overload of SetParameterPointer as well for all
  // non-structured data?
  LOG_DEBUG("Registered Parameter");
  if (*ier) return;

  // Register function pointers
  // -----------------------------------------------------------
  registerFunctionPointers(modelDriverCreate, ier);
  if (*ier) return;

  // Set
  // Units----------------------------------------------------------------------------
  *ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                     requestedEnergyUnit,
                                     KIM::CHARGE_UNIT::unused,
                                     KIM::TEMPERATURE_UNIT::unused,
                                     KIM::TIME_UNIT::unused);

  // Set preprocessor descriptor callbacks
  // --------------------------------------------------
  if (preprocessing == "descriptor")
  {
#ifdef USE_LIBDESC
    descriptor = std::unique_ptr<Descriptor::DescriptorKind>(
        DescriptorKind::initDescriptor(descriptor_param_file, descriptor_kind));
#else
    LOG_ERROR("Descriptor preprocessing is not supported in this build");
    *ier = true;
    return;
#endif
  }
  else if (preprocessing == "graph")
  {
#ifndef DISABLE_GRAPH
    for (int i = 0; i < n_layers; i++)
    {
      graph_edge_indices.push_back(std::vector<std::int64_t> {});
      // graph_edge_indices[i].assign(std::pow(2 * MAX_NEIGHBORS, i + 1), -1);
      // exponential memory allocation, might get out of hand
    }
#ifdef USE_LIBDESC
    descriptor = nullptr;
#endif
#else
    LOG_ERROR("Graph preprocessing is not supported in this build");
    *ier = true;
    return;
#endif
  }
}

//******************************************************************************
// TODO: Can be done with templating. Deal with it later
int TorchMLModelDriverImplementation::Refresh(
    KIM::ModelRefresh * const modelRefresh)
{
  modelRefresh->SetInfluenceDistancePointer(&influence_distance);
  modelRefresh->SetNeighborListPointers(
      1,
      &cutoff_distance,
      &modelWillNotRequestNeighborsOfNoncontributingParticles_);
  // As all param are part of torch model, nothing to do here?
  // TODO Distance matrix for computational efficiency, which will be refreshed
  // to -1
  return false;
}

int TorchMLModelDriverImplementation::Refresh(
    KIM::ModelDriverCreate * const modelRefresh)
{
  modelRefresh->SetInfluenceDistancePointer(&influence_distance);
  modelRefresh->SetNeighborListPointers(
      1,
      &cutoff_distance,
      &modelWillNotRequestNeighborsOfNoncontributingParticles_);
  // TODO Distance matrix for computational efficiency, which will be refreshed
  // to -1
  return false;
}

//******************************************************************************
int TorchMLModelDriverImplementation::Compute(
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  Run(modelComputeArguments);
  // TODO see proper way to return error codes
  return false;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate

int TorchMLModelDriverImplementation::ComputeArgumentsCreate(
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
{
  LOG_INFORMATION("Compute argument create");
  int error = modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
                  KIM::SUPPORT_STATUS::required)
              // Support state can be optional for energy
              // But then we need to explicitly handle nullptr case
              // As energy is anyway always needs to be computed
              // as a prerequisite for force computation, easier to
              // make it required.
              // TODO: Handle it properly in future for GD models
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialForces,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
                  KIM::SUPPORT_STATUS::notSupported);
  // register callbacks
  error = error
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
              KIM::SUPPORT_STATUS::notSupported)
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
              KIM::SUPPORT_STATUS::notSupported);
  return error;
  LOG_INFORMATION("Register callback supportStatus");
}

// *****************************************************************************
// Auxiliary methods------------------------------------------------------------

void TorchMLModelDriverImplementation::Run(
    const KIM::ModelComputeArguments * const modelComputeArguments)
{
  contributingAtomCounts(modelComputeArguments);
  preprocessInputs(modelComputeArguments);
  postprocessOutputs(modelComputeArguments);
}

// -----------------------------------------------------------------------------
void TorchMLModelDriverImplementation::preprocessInputs(
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  // TODO: Make preprocessing type enums
  if (preprocessing == "none") { setDefaultInputs(modelComputeArguments); }
  else if (preprocessing == "descriptor")
  {
#ifdef USE_LIBDESC
    setDescriptorInputs(modelComputeArguments);
#endif
  }
  else if (preprocessing == "graph")
  {
#ifndef DISABLE_GRAPH
    setGraphInputs(modelComputeArguments);
#endif
  }
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::postprocessOutputs(
    KIM::ModelComputeArguments const * modelComputeArguments)
{
  double * energy = nullptr;
  double * partialEnergy = nullptr;
  double * forces = nullptr;
  int * particleSpeciesCodes;  // FIXME: Implement species code handling
  int const * numberOfParticlesPointer;
  int const * particleContributing = nullptr;
  double * coordinates = nullptr;
  // double *virial = nullptr; // Not supported yet

  auto ier
      = modelComputeArguments->GetArgumentPointer(
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
            const_cast<double const **>(
                &coordinates))  // Needed for libdescriptor
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
            static_cast<double **>(&forces))
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy, &energy)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy, &partialEnergy);
  //  || modelComputeArguments->GetArgumentPointer(
  //   KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
  //   &virial);
  if (ier) return;

  if (preprocessing != "descriptor")
  {
    if (*numberOfParticlesPointer == 1)
    {  // padded single particle GNN
      auto forces_padded
          = std::make_unique<double[]>((*numberOfParticlesPointer + 1) * 3);
      ml_model->Run(
          energy, partialEnergy, forces_padded.get(), !returns_forces);
      std::memcpy(forces,
                  forces_padded.get(),
                  *numberOfParticlesPointer * 3 * sizeof(double));
    }
    else { ml_model->Run(energy, partialEnergy, forces, !returns_forces); }
  }
  else
  {  // descriptor if-else
#ifdef USE_LIBDESC
    auto neg_dE_dzeta
        = std::make_unique<double[]>(n_contributing_atoms * descriptor->width);

    ml_model->Run(energy, partialEnergy, neg_dE_dzeta.get(), !returns_forces);
    // only do below if forces are needed
    if (forces && neg_dE_dzeta)
    {  // forces were requested and model returned valid grad
      int neigh_from = 0;
      int n_neigh;
      int width = descriptor->width;
      int contributing_particle_ptr = 0;
      // allocate memory to access forces, and give the location to
      auto force_accessor
          = std::make_unique<double[]>(*numberOfParticlesPointer * 3);
      std::fill(force_accessor.get(),
                force_accessor.get() + *numberOfParticlesPointer * 3,
                0.0);
      for (int i = 0; i < n_contributing_atoms; i++)
      {
        if (particleContributing[i] != 1) { continue; }
        n_neigh = num_neighbors_[contributing_particle_ptr];
        std::vector<int> n_list(neighbor_list.begin() + neigh_from,
                                neighbor_list.begin() + neigh_from + n_neigh);
        neigh_from += n_neigh;
        // Single atom gradient from descriptor
        // TODO: call gradient function, which handles atom-wise iteration
        gradient_single_atom(
            i,
            *numberOfParticlesPointer,
            particleSpeciesCodes,
            n_list.data(),
            n_neigh,
            coordinates,
            force_accessor.get(),
            descriptor_array.data() + (contributing_particle_ptr * width),
            neg_dE_dzeta.get() + (contributing_particle_ptr * width),
            descriptor.get());
        contributing_particle_ptr++;
      }
      // neg_dE_dzeta = -dE/dzeta, therefore forces = neg_d_desc * dzeta/dr, no
      // negation needed now
      std::memcpy(forces,
                  force_accessor.get(),
                  *numberOfParticlesPointer * 3 * sizeof(double));
    }
    else if (forces && !neg_dE_dzeta)
    {
      // something is wrong if forces are requested but input_grad is not
      // available
      LOG_ERROR("Forces requested but model did not provide valid gradient");
      return;
    }
#endif
  }  // descriptor if else
}

// -----------------------------------------------------------------------------
void TorchMLModelDriverImplementation::updateNeighborList(
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  int const * numberOfParticlesPointer = nullptr;
  int const * particleContributing = nullptr;
  auto ier = modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
                 &numberOfParticlesPointer)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
                 &particleContributing);
  if (ier)
  {
    LOG_ERROR(
        "Could not create model compute arguments input @ updateNeighborList");
    return;
  }
  int numOfNeighbors;
  int const * neighbors;
  num_neighbors_.clear();
  neighbor_list.clear();
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (particleContributing[i] == 1)
    {
      modelComputeArguments->GetNeighborList(0, i, &numOfNeighbors, &neighbors);
      num_neighbors_.push_back(numOfNeighbors);
      for (int neigh = 0; neigh < numOfNeighbors; neigh++)
      {
        neighbor_list.push_back(neighbors[neigh]);
      }
    }
  }
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::setDefaultInputs(
    const KIM::ModelComputeArguments * modelComputeArguments)
{
  int const * numberOfParticlesPointer;
  int *
      particleSpeciesCodes;  // FIXME: Implement species code handling Ask Ryan
  int * particleContributing = nullptr;
  double * coordinates = nullptr;

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
                 const_cast<double const **>(&coordinates));
  if (ier)
  {
    LOG_ERROR(
        "Could not create model compute arguments input @ setDefaultInputs");
    return;
  }

  updateNeighborList(modelComputeArguments);

  species_atomic_number.assign(*numberOfParticlesPointer, 0);
  contraction_array.assign(*numberOfParticlesPointer, 0);

  bool map_species_z = std::getenv("KIM_MODEL_ELEMENTS_MAP") != nullptr;

  if (map_species_z)
  {
    for (int i = 0; i < *numberOfParticlesPointer; i++)
    {
      species_atomic_number[i] = z_map[particleSpeciesCodes[i]];
      contraction_array[i] = particleContributing[i];
    }
  }
  else
  {
    for (int i = 0; i < *numberOfParticlesPointer; i++)
    {
      species_atomic_number[i] = particleSpeciesCodes[i];
      contraction_array[i] = particleContributing[i];
    }
  }

  auto shape = std::vector<std::int64_t> {*numberOfParticlesPointer};
  ml_model->SetInputNode(0, species_atomic_number.data(), shape, false, true);


  shape.clear();
  shape = {*numberOfParticlesPointer, 3};

  ml_model->SetInputNode(1, coordinates, shape, true, true);


  shape.clear();
  shape = {static_cast<std::int64_t>(num_neighbors_.size())};
  ml_model->SetInputNode(2, num_neighbors_.data(), shape, false, true);

  shape.clear();
  shape = {static_cast<std::int64_t>(neighbor_list.size())};

  ml_model->SetInputNode(3, neighbor_list.data(), shape, false, true);

  shape.clear();
  shape = {*numberOfParticlesPointer};

  ml_model->SetInputNode(4, contraction_array.data(), shape, false, true);
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::setDescriptorInputs(
    const KIM::ModelComputeArguments * modelComputeArguments)
{
  int const * numberOfParticlesPointer;
  int * particleSpeciesCodes;  // FIXME: Implement species code handling
  int * particleContributing = nullptr;
  double * coordinates = nullptr;
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
                 const_cast<double const **>(&coordinates));
  if (ier)
  {
    LOG_ERROR(
        "Could not create model compute arguments input @ setDefaultInputs");
    return;
  }
#ifdef USE_LIBDESC
  int neigh_from, n_neigh;
  neigh_from = 0;
  int width = descriptor->width;
  updateNeighborList(modelComputeArguments);

  descriptor_array.assign(n_contributing_atoms * width, 0.0);

  int contributing_particle_ptr = 0;
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (particleContributing[i] != 1) { continue; }
    n_neigh = num_neighbors_[contributing_particle_ptr];
    std::vector<int> n_list(neighbor_list.begin() + neigh_from,
                            neighbor_list.begin() + neigh_from + n_neigh);
    neigh_from += n_neigh;
    // Single atom descriptor wrapper from descriptor
    // TODO: call compute function, which handles atom-wise iteration
    compute_single_atom(i,
                        *numberOfParticlesPointer,
                        particleSpeciesCodes,
                        n_list.data(),
                        n_neigh,
                        coordinates,
                        descriptor_array.data()
                            + (contributing_particle_ptr * width),
                        descriptor.get());
    contributing_particle_ptr++;
  }

  std::vector<std::int64_t> shape({n_contributing_atoms, width});
  ml_model->SetInputNode(0, descriptor_array.data(), shape, true, true);
#else
  throw std::runtime_error("Descriptor not compiled in; this should not have "
                           "executed. Please report this bug.");
#endif
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::setGraphInputs(
    const KIM::ModelComputeArguments * modelComputeArguments)
{
  int const * numberOfParticlesPointer;
  int * particleSpeciesCodes;  // FIXME: Implement species code handling
  int * particleContributing = nullptr;
  double * coordinates = nullptr;
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
                 const_cast<double const **>(&coordinates));
  if (ier)
  {
    LOG_ERROR(
        "Could not create model compute arguments input @ setDefaultInputs");
    return;
  }
  int numberOfNeighbors;
  int const * neighbors;

  std::unordered_set<int> atoms_in_layers;
  std::vector<std::unordered_set<std::array<std::int64_t, 2>, CantorPairing> >
      staged_graph(n_layers);

  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (particleContributing[i] == 1) { atoms_in_layers.insert(i); }
  }

  double cutoff_sq = cutoff_distance * cutoff_distance;
  double _x, _y, _z;
  std::array<double, 3> i_arr = {0.0, 0.0, 0.0}, j_arr = {0.0, 0.0, 0.0};
  int i_layer = 0;
  constexpr int data_copy_size_arr3 = 3 * sizeof(double);

  do {
    std::unordered_set<int> atoms_in_next_layer;

    // for each atom in current layer, append its edges in graph
    // and collect the neighbors for next convolution step
    for (int atom_i : atoms_in_layers)
    {
      // TODO: reuse the neighbor lists? rather than calling GetNeighborList
      // I think it is a shallow pointer reference so it might not help much

      modelComputeArguments->GetNeighborList(
          0, atom_i, &numberOfNeighbors, &neighbors);
      for (int j = 0; j < numberOfNeighbors; j++)
      {
        int atom_j = neighbors[j];
        std::memcpy(
            i_arr.data(), coordinates + 3 * atom_i, data_copy_size_arr3);
        std::memcpy(
            j_arr.data(), coordinates + 3 * atom_j, data_copy_size_arr3);
        _x = j_arr[0] - i_arr[0];
        _y = j_arr[1] - i_arr[1];
        _z = j_arr[2] - i_arr[2];
        double r_sq = _x * _x + _y * _y + _z * _z;
        if (r_sq <= cutoff_sq)
        {
          staged_graph[i_layer].insert({atom_i, atom_j});
          staged_graph[i_layer].insert({atom_j, atom_i});
          atoms_in_next_layer.insert(atom_j);
        }
      }
    }
    atoms_in_layers = atoms_in_next_layer;
    i_layer++;
  } while (i_layer < n_layers);


  i_layer = 0;
  for (auto const & edge_index_set : staged_graph)
  {
    int jj = 0;
    auto single_graph_size = edge_index_set.size();
    // Sanitize previous graph
    graph_edge_indices[i_layer].assign(single_graph_size * 2, -1);
    for (auto & bond_pair : edge_index_set)
    {
      graph_edge_indices[i_layer][jj] = std::get<0>(bond_pair);
      graph_edge_indices[i_layer][jj + single_graph_size]
          = std::get<1>(bond_pair);
      jj++;
    }
    i_layer++;
  }

  species_atomic_number.assign(*numberOfParticlesPointer, 0);

  // TODO: Read this from file
  //  Get environment variable KIM_MODEL_ELEMENTS_MAP, and set map_species_z to
  //  true if it is set, else false
  bool map_species_z = std::getenv("KIM_MODEL_ELEMENTS_MAP") != nullptr;

  if (map_species_z)
  {
    for (int i = 0; i < *numberOfParticlesPointer; i++)
    {
      species_atomic_number[i] = z_map[particleSpeciesCodes[i]];
    }
  }
  else
  {
    for (int i = 0; i < *numberOfParticlesPointer; i++)
    {
      species_atomic_number[i] = particleSpeciesCodes[i];
    }
  }

  contraction_array.assign(*numberOfParticlesPointer, 1);
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    contraction_array[i] = (particleContributing[i] == 0) ? 1 : 0;
  }

  // Fix for isolated atoms. Append a dummy particle, not in graph
  std::unique_ptr<double[]> padded_coordinates;
  int effectiveNumberOfParticlePointers;
  auto shape = std::vector<std::int64_t> {};

  if (*numberOfParticlesPointer == 1)
  {
    effectiveNumberOfParticlePointers
        = (*numberOfParticlesPointer == 1) ? 2 : *numberOfParticlesPointer;
    species_atomic_number.push_back(particleSpeciesCodes[0]);
    contraction_array.push_back(1);
    padded_coordinates = std::make_unique<double[]>((*numberOfParticlesPointer + 1) * 3);
    std::memcpy(padded_coordinates.get(),
                coordinates,
                *numberOfParticlesPointer * 3 * sizeof(double));
    padded_coordinates[*numberOfParticlesPointer * 3 + 0] = 999.0;
    padded_coordinates[*numberOfParticlesPointer * 3 + 1] = 999.0;
    padded_coordinates[*numberOfParticlesPointer * 3 + 2] = 999.0;
    shape.clear();
    shape = {effectiveNumberOfParticlePointers, 3};
    ml_model->SetInputNode(1, padded_coordinates.get(), shape, true, true);
  }
  else
  {
    effectiveNumberOfParticlePointers = *numberOfParticlesPointer;
    shape.clear();
    shape = {effectiveNumberOfParticlePointers, 3};
    ml_model->SetInputNode(1, coordinates, shape, true, true);
  }
  shape.clear();
  shape = {effectiveNumberOfParticlePointers};

  ml_model->SetInputNode(0, species_atomic_number.data(), shape, false, true);


  for (int i = 0; i < n_layers; i++)
  {
    shape.clear();
    shape = {2, static_cast<std::int64_t>(graph_edge_indices[i].size() / 2)};
    ml_model->SetInputNode(
        2 + i, graph_edge_indices[i].data(), shape, false, true);
  }

  shape.clear();
  shape = {effectiveNumberOfParticlePointers};
  ml_model->SetInputNode(
      2 + n_layers, contraction_array.data(), shape, false, true);
}

// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchMLModelDriverImplementation::readParametersFile(
    KIM::ModelDriverCreate * modelDriverCreate, int * ier)
{
  // Read parameter files from model driver
  // ---------------------------------------
  int num_param_files;
  std::string const *param_file_name, *tmp_file_name;
  std::string const * param_dir_name;
  std::string const * model_file_name;
  std::string const * descriptor_file_name;

  modelDriverCreate->GetNumberOfParameterFiles(&num_param_files);

  // Only 2 files expected .param and .pt
  if (num_param_files > MAX_FILE_NUM)
  {
    *ier = true;
    LOG_ERROR("Too many parameter files");
    return;
  }

  for (int i = 0; i < num_param_files; i++)
  {
    modelDriverCreate->GetParameterFileBasename(i, &tmp_file_name);
    if (tmp_file_name->substr(tmp_file_name->size() - 5) == "param")
    {
      param_file_name = tmp_file_name;
    }
    else if (tmp_file_name->substr(tmp_file_name->size() - 2) == "pt")
    {
      model_file_name = tmp_file_name;
    }
    else if (tmp_file_name->substr(tmp_file_name->size() - 3) == "dat")
    {
      descriptor_file_name = tmp_file_name;
    }
    else
    {
      LOG_ERROR("File extensions do not match; only expected .param or .pt");
      *ier = true;
      return;
    }
  }

  // Get param directory to load model and parameters from
  modelDriverCreate->GetParameterFileDirectoryName(&param_dir_name);

  std::string full_qualified_file_name
      = *param_dir_name + "/" + *param_file_name;
  fully_qualified_model_name = *param_dir_name + "/" + *model_file_name;

  std::string placeholder_string;

  std::fstream file_ptr(full_qualified_file_name);

  if (file_ptr.is_open())
  {
    // TODO better structured input block. YAML?
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');

    n_elements = std::stoi(placeholder_string);
    std::getline(file_ptr, placeholder_string);

    for (int i = 0; i < n_elements; i++)
    {
      auto pos = placeholder_string.find(' ');
      elements_list.push_back(placeholder_string.substr(0, pos));
      if (pos == std::string::npos)
      {
        if (i + 1 != n_elements)
        {
          LOG_ERROR("Incorrect formatting OR number of elements");
          *ier = true;
          return;
        }
        LOG_DEBUG("Number of elements read: " + std::to_string(i + 1));
      }
      else { placeholder_string.erase(0, pos + 1); }
    }
    // Species Z map
    for (int i = 0; i < n_elements; i++)
    {
      z_map.push_back(sym_to_z(elements_list[i]));
    }

    // blank line
    std::getline(file_ptr, placeholder_string);
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');
    // which preprocessing to use
    preprocessing = placeholder_string;
    std::transform(preprocessing.begin(),
                   preprocessing.end(),
                   preprocessing.begin(),
                   ::tolower);

    // blank line
    std::getline(file_ptr, placeholder_string);
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');
    // influence distance
    cutoff_distance = std::stod(placeholder_string);
    n_layers = 0;
    if (preprocessing == "graph")
    {
#ifndef DISABLE_GRAPH
      std::getline(file_ptr, placeholder_string);
      n_layers = std::stoi(placeholder_string);
      influence_distance = cutoff_distance * n_layers;
#else
      LOG_ERROR("Graph preprocessing not supported");
      *ier = true;
      return;
#endif
    }
    else { influence_distance = cutoff_distance; }

    // blank line
    std::getline(file_ptr, placeholder_string);
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');
    // Model name for comparison
    model_name = placeholder_string;


    // blank line
    std::getline(file_ptr, placeholder_string);
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');
    // Does the model return forces? If no then we need to compute gradients
    // If yes we can optimize it further using inference mode

    auto return_forces_str = std::string{placeholder_string};
    std::transform(return_forces_str.begin(), return_forces_str.end(), return_forces_str.begin(), ::tolower);
    returns_forces = return_forces_str == "true";

    // blank line
    std::getline(file_ptr, placeholder_string);
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');
    // number of strings
    number_of_inputs = std::stoi(placeholder_string);

    if (preprocessing == "descriptor")
    {
#ifdef USE_LIBDESC
      // blank line
      std::getline(file_ptr, placeholder_string);
      // Ignore comments
      do {
        std::getline(file_ptr, placeholder_string);
      } while (placeholder_string[0] == '#');
      // number of strings
      descriptor_name = placeholder_string;
      std::transform(descriptor_name.begin(), descriptor_name.end(), descriptor_name.begin(), ::tolower);
      descriptor_param_file = *param_dir_name + "/" + *descriptor_file_name;
      if (descriptor_name == "symmetryfunctions")
      {
        descriptor_kind = AvailableDescriptor::KindSymmetryFunctions;
      }
      else if (descriptor_name == "bispectrum")
      {
        descriptor_kind = AvailableDescriptor::KindBispectrum;
      }
      else if (descriptor_name == "soap")
      {
        descriptor_kind = AvailableDescriptor::KindSOAP;
      }
      else { throw std::invalid_argument("Descriptor not supported."); }
      std::ifstream desc_file(descriptor_param_file, std::ios::in);
      if (!desc_file) { throw std::runtime_error("Descriptor file not found"); }
      std::stringstream buffer;
      buffer << desc_file.rdbuf();
      descriptor_param_file_content = buffer.str();
#else
      LOG_ERROR("Descriptor preprocessing requires libdescriptor");
      *ier = true;
      return;
#endif
    }
  }
  else
  {
    LOG_ERROR("Param file not found");
    *ier = true;
    return;
  }
  file_ptr.close();

  LOG_DEBUG("Successfully parsed parameter file");
  if (*model_file_name != model_name)
  {
    LOG_ERROR("Provided model file name different from present model file.");
    *ier = true;
    return;
  }
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelWriteParameterizedModel
int TorchMLModelDriverImplementation::WriteParameterizedModel(
    const KIM::ModelWriteParameterizedModel * const
        modelWriteParameterizedModel) const
{
  std::string buffer;
  std::string const * path;
  std::string const * modelName;

  modelWriteParameterizedModel->GetPath(&path);
  modelWriteParameterizedModel->GetModelName(&modelName);

  buffer = *modelName + ".params";
  modelWriteParameterizedModel->SetParameterFileName(buffer);

  buffer = *path + "/" + *modelName + ".params";
  std::ofstream fp(buffer.c_str());
  if (!fp.is_open())
  {
    LOG_ERROR("Unable to open parameter file for writing.");
    return true;
  }
  fp << "# Num of elements\n";
  fp << n_elements << "\n";
  for (auto & elem : elements_list) { fp << elem << " "; }
  fp << "\n";
  fp << "# preprocessing\n";
  fp << cutoff_distance << "\n";

  if (preprocessing == "graph") { fp << n_layers << "\n\n"; }

  fp << "# Model name\n";
  fp << model_name << "\n\n";

  fp << "# Return forces\n";
  fp << (returns_forces ? "True" : "False") << "\n\n";

  fp << "# Number of inputs\n";
  fp << number_of_inputs << "\n\n";

  fp << "# Descriptor, if any\n";
  fp << descriptor_name;

  fp.close();

  if (preprocessing == "descriptor")
  {
    std::string descriptor_file = *path + "/" + "descriptor.dat";
    std::ofstream fp_desc(descriptor_file);
    if (!fp_desc.is_open())
    {
      LOG_ERROR("Unable to open descriptor parameter file");
    }
    fp_desc << descriptor_param_file_content;
    fp_desc.close();
  }

  // model file
  std::string ml_model_file = *path + "/" + model_name;
  ml_model->WriteMLModel(ml_model_file);  // torch will handle io issues

  // CMAKEFILES?

  LOG_INFORMATION("Saved model to disk");
  return false;
}


// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchMLModelDriverImplementation::unitConversion(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    [[maybe_unused]]KIM::ChargeUnit const requestedChargeUnit,
    [[maybe_unused]]KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
  // KIM::LengthUnit fromLength = KIM::LENGTH_UNIT::A;
  // KIM::EnergyUnit fromEnergy = KIM::ENERGY_UNIT::eV;
  // KIM::ChargeUnit fromCharge = KIM::CHARGE_UNIT::e;
  // KIM::TemperatureUnit fromTemperature = KIM::TEMPERATURE_UNIT::K;
  // KIM::TimeUnit fromTime = KIM::TIME_UNIT::ps;
  // double convertLength = 1.0;
  if (requestedLengthUnit != KIM::LENGTH_UNIT::A)
  {
    LOG_ERROR("Only Angstroms supported for length unit");
    *ier = true;
    return;
  }
  if (requestedEnergyUnit != KIM::ENERGY_UNIT::eV)
  {
    LOG_ERROR("Only eV supported for energy unit");
    *ier = true;
    return;
  }

  *ier = modelDriverCreate->SetUnits(KIM::LENGTH_UNIT::A,
                                     KIM::ENERGY_UNIT::eV,
                                     KIM::CHARGE_UNIT::unused,
                                     KIM::TEMPERATURE_UNIT::unused,
                                     requestedTimeUnit);
}

// --------------------------------------------------------------------------------
void TorchMLModelDriverImplementation::setSpecies(
    KIM::ModelDriverCreate * const modelDriverCreate, int * const ier)
{
  // This one is bit confusing, currently ML model driver supports two modes
  // 1. use zero based indices for species, most common
  // 2. provide atomic numbers
  // The zero based species is hard baked, as all ML models provide single index
  // for species (basically a proxy for Z), by that logic, the simple assignment
  // below is correct. But still need to consult Ryan and Ilia once.
  // TODO: ensure species assignment is correct.
  int code = 0;
  for (auto const & species : elements_list)
  {
    KIM::SpeciesName const specName(species);
    *ier = modelDriverCreate->SetSpeciesCode(specName, code);
    code += 1;
    if (*ier) return;
  }
}

// --------------------------------------------------------------------------------
void TorchMLModelDriverImplementation::registerFunctionPointers(
    KIM::ModelDriverCreate * const modelDriverCreate, int * const ier)
{
  // Use function pointer definitions to verify correct prototypes
  // TODO This doesn't look nice, implementation calling parent class
  // See if there is a workaround
  KIM::ModelDestroyFunction * destroy = TorchMLModelDriver::Destroy;
  KIM::ModelRefreshFunction * refresh = TorchMLModelDriver::Refresh;
  KIM::ModelComputeFunction * compute = TorchMLModelDriver::Compute;
  KIM::ModelComputeArgumentsCreateFunction * CACreate
      = TorchMLModelDriver::ComputeArgumentsCreate;
  KIM::ModelComputeArgumentsDestroyFunction * CADestroy
      = TorchMLModelDriver::ComputeArgumentsDestroy;

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
int TorchMLModelDriverImplementation::ComputeArgumentsDestroy(
    KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy)
{
  // Nothing to do here?
  // As per my understanding, the ComputeArgumentsDestroy is used for
  // de-allocation and destruction of array compute arguments. We have no such
  // allocation
  TorchMLModelDriver * modelObject;  // To silence the compiler
  modelComputeArgumentsDestroy->GetModelBufferPointer(
      reinterpret_cast<void **>(&modelObject));
  return false;
}


//-------------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchMLModelDriverImplementation::contributingAtomCounts(
    const KIM::ModelComputeArguments * modelComputeArguments)
{
  int * numberOfParticlesPointer;
  int * particleContributing;
  auto ier = modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
                 &numberOfParticlesPointer)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
                 &particleContributing);
  if (ier)
  {
    LOG_ERROR("Could not get number of particles @ contributingAtomCount");
    return;
  }
  n_contributing_atoms = 0;
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (particleContributing[i] == 1) { n_contributing_atoms += 1; }
  }
}


// *****************************************************************************
TorchMLModelDriverImplementation::~TorchMLModelDriverImplementation() = default;


// *****************************************************************************
int sym_to_z(std::string & sym)
{
  // TODO more idiomatic handling of species. Ask Ryan
  static const std::unordered_map<std::string, int> element_map
      = {{"H", 1},   {"He", 2},  {"Li", 3},  {"Be", 4},  {"B", 5},   {"C", 6},
         {"N", 7},   {"O", 8},   {"F", 9},   {"Ne", 10}, {"Na", 11}, {"Mg", 12},
         {"Al", 13}, {"Si", 14}, {"P", 15},  {"S", 16},  {"Cl", 17}, {"A", 18},
         {"K", 19},  {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23},  {"Cr", 24},
         {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
         {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36},
         {"Rb", 37}, {"Sr", 38}, {"Y", 39},  {"Zr", 40}, {"Nb", 41}, {"Mo", 42},
         {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48},
         {"In", 49}, {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53},  {"Xe", 54},
         {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60},
         {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66},
         {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71}, {"Hf", 72},
         {"Ta", 73}, {"W", 74},  {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78},
         {"Au", 79}, {"Hg", 80}, {"Ti", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84},
         {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90},
         {"Pa", 91}, {"U", 92}};
  auto it = element_map.find(sym);
  return (it != element_map.end()) ? it->second : -1;
}