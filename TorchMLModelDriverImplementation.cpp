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
  auto device = std::string {std::getenv("KIM_MODEL_EXECUTION_DEVICE")};
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
  if (preprocessing == "Graph")
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
  if (preprocessing == "Descriptor")
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
  else if (preprocessing == "Graph")
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
  if (preprocessing == "None") { setDefaultInputs(modelComputeArguments); }
  else if (preprocessing == "Descriptor")
  {
#ifdef USE_LIBDESC
    setDescriptorInputs(modelComputeArguments);
#endif
  }
  else if (preprocessing == "Graph")
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

  if (preprocessing != "Descriptor")
  {
    ml_model->Run(energy, partialEnergy, forces, !returns_forces);
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
      std::memcpy(forces, force_accessor.get(), *numberOfParticlesPointer * 3);
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
    KIM::ModelComputeArguments const * const modelComputeArguments,
    int const numberOfParticles)
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

  updateNeighborList(modelComputeArguments, n_contributing_atoms);

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
  updateNeighborList(modelComputeArguments, n_contributing_atoms);

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
  std::vector<std::unordered_set<std::array<std::int64_t, 2>,
                                 SymmetricCantorPairing,
                                 SymmetricPairEqual> >
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
    graph_edge_indices[i_layer].assign(single_graph_size * 4, -1);
    for (auto & bond_pair : edge_index_set)
    {
      graph_edge_indices[i_layer][jj] = std::get<0>(bond_pair);
      graph_edge_indices[i_layer][jj + 2 * single_graph_size]
          = std::get<1>(bond_pair);
      jj++;
    }
    std::memcpy(graph_edge_indices[i_layer].data() + single_graph_size,
                graph_edge_indices[i_layer].data() + 2 * single_graph_size,
                single_graph_size * sizeof(std::int64_t));
    std::memcpy(graph_edge_indices[i_layer].data() + 3 * single_graph_size,
                graph_edge_indices[i_layer].data(),
                single_graph_size * sizeof(std::int64_t));
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

  // Fix for isolated atoms. Append a dummy particle, not in graph
  int effectiveNumberOfParticlePointers
      = (*numberOfParticlesPointer == 1) ? 2 : *numberOfParticlesPointer;

  auto shape = std::vector<std::int64_t> {effectiveNumberOfParticlePointers};
  ml_model->SetInputNode(0, species_atomic_number.data(), shape, false, true);

  shape.clear();
  shape = {effectiveNumberOfParticlePointers, 3};
  ml_model->SetInputNode(1, coordinates, shape, true, true);

  for (int i = 0; i < n_layers; i++)
  {
    shape.clear();
    shape = {2, static_cast<std::int64_t>(graph_edge_indices[i].size() / 2)};
    ml_model->SetInputNode(
        2 + i, graph_edge_indices[i].data(), shape, false, true);
  }

  contraction_array.assign(*numberOfParticlesPointer, 0);
  for (int i = 0; i < effectiveNumberOfParticlePointers; i++)
  {
    contraction_array[i] = (particleContributing[i] == 0) ? 1 : 0;
  }
  shape.clear();
  shape = {*numberOfParticlesPointer};
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
    // std::transform(preprocessing.begin(),preprocessing.end(),
    //                preprocessing.begin(),::tolower);

    // blank line
    std::getline(file_ptr, placeholder_string);
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');
    // influence distance
    cutoff_distance = std::stod(placeholder_string);
    n_layers = 0;
    if (preprocessing == "Graph")
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
    for (char & t : placeholder_string) t = static_cast<char>(tolower(t));
    returns_forces
        = placeholder_string == "true" || placeholder_string == "True";

    // blank line
    std::getline(file_ptr, placeholder_string);
    // Ignore comments
    do {
      std::getline(file_ptr, placeholder_string);
    } while (placeholder_string[0] == '#');
    // number of strings
    number_of_inputs = std::stoi(placeholder_string);

    if (preprocessing == "Descriptor")
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
      descriptor_param_file = *param_dir_name + "/" + *descriptor_file_name;
      if (descriptor_name == "SymmetryFunctions")
      {
        descriptor_kind = AvailableDescriptor::KindSymmetryFunctions;
      }
      else if (descriptor_name == "Bispectrum")
      {
        descriptor_kind = AvailableDescriptor::KindBispectrum;
      }
      else if (descriptor_name == "SOAP")
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

  if (preprocessing == "Graph") { fp << n_layers << "\n\n"; }

  fp << "# Model name\n";
  fp << model_name << "\n\n";

  fp << "# Return forces\n";
  fp << (returns_forces ? "True" : "False") << "\n\n";

  fp << "# Number of inputs\n";
  fp << number_of_inputs << "\n\n";

  fp << "# Descriptor, if any\n";
  fp << descriptor_name;

  fp.close();

  if (preprocessing == "Descriptor")
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
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
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
  if (sym == "H") return 1;
  if (sym == "He") return 2;
  if (sym == "Li") return 3;
  if (sym == "Be") return 4;
  if (sym == "B") return 5;
  if (sym == "C") return 6;
  if (sym == "N") return 7;
  if (sym == "O") return 8;
  if (sym == "F") return 9;
  if (sym == "Ne") return 10;
  if (sym == "Na") return 11;
  if (sym == "Mg") return 12;
  if (sym == "Al") return 13;
  if (sym == "Si") return 14;
  if (sym == "P") return 15;
  if (sym == "S") return 16;
  if (sym == "Cl") return 17;
  if (sym == "A") return 18;
  if (sym == "K") return 19;
  if (sym == "Ca") return 20;
  if (sym == "Sc") return 21;
  if (sym == "Ti") return 22;
  if (sym == "V") return 23;
  if (sym == "Cr") return 24;
  if (sym == "Mn") return 25;
  if (sym == "Fe") return 26;
  if (sym == "Co") return 27;
  if (sym == "Ni") return 28;
  if (sym == "Cu") return 29;
  if (sym == "Zn") return 30;
  if (sym == "Ga") return 31;
  if (sym == "Ge") return 32;
  if (sym == "As") return 33;
  if (sym == "Se") return 34;
  if (sym == "Br") return 35;
  if (sym == "Kr") return 36;
  if (sym == "Rb") return 37;
  if (sym == "Sr") return 38;
  if (sym == "Y") return 39;
  if (sym == "Zr") return 40;
  if (sym == "Nb") return 41;
  if (sym == "Mo") return 42;
  if (sym == "Tc") return 43;
  if (sym == "Ru") return 44;
  if (sym == "Rh") return 45;
  if (sym == "Pd") return 46;
  if (sym == "Ag") return 47;
  if (sym == "Cd") return 48;
  if (sym == "In") return 49;
  if (sym == "Sn") return 50;
  if (sym == "Sb") return 51;
  if (sym == "Te") return 52;
  if (sym == "I") return 53;
  if (sym == "Xe") return 54;
  if (sym == "Cs") return 55;
  if (sym == "Ba") return 56;
  if (sym == "La") return 57;
  if (sym == "Ce") return 58;
  if (sym == "Pr") return 59;
  if (sym == "Nd") return 60;
  if (sym == "Pm") return 61;
  if (sym == "Sm") return 62;
  if (sym == "Eu") return 63;
  if (sym == "Gd") return 64;
  if (sym == "Tb") return 65;
  if (sym == "Dy") return 66;
  if (sym == "Ho") return 67;
  if (sym == "Er") return 68;
  if (sym == "Tm") return 69;
  if (sym == "Yb") return 70;
  if (sym == "Lu") return 71;
  if (sym == "Hf") return 72;
  if (sym == "Ta") return 73;
  if (sym == "W") return 74;
  if (sym == "Re") return 75;
  if (sym == "Os") return 76;
  if (sym == "Ir") return 77;
  if (sym == "Pt") return 78;
  if (sym == "Au") return 79;
  if (sym == "Hg") return 80;
  if (sym == "Ti") return 81;
  if (sym == "Pb") return 82;
  if (sym == "Bi") return 83;
  if (sym == "Po") return 84;
  if (sym == "At") return 85;
  if (sym == "Rn") return 86;
  if (sym == "Fr") return 87;
  if (sym == "Ra") return 88;
  if (sym == "Ac") return 89;
  if (sym == "Th") return 90;
  if (sym == "Pa") return 91;
  if (sym == "U") return 92;
  return -1;
}
