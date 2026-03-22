#include "MLModel.hpp"
#include "TorchExportModel.hpp"
#include "TorchScriptModel.hpp"

std::unique_ptr<MLModel> CreateModel(MLModelTypes type,
                                     std::string & fully_qualified_model_name,
                                     std::string & device,
                                     int number_of_inputs)
{
  switch (type)
  {
    case TORCHEXPORT:
      return std::make_unique<TorchExportModel>(fully_qualified_model_name,device,number_of_inputs);
    case TORCHSCRIPT:
      return std::make_unique<TorchScriptModel>(fully_qualified_model_name,device,number_of_inputs);
    default:
      return nullptr;
  }
}