#ifndef __deal2__cdr_1_parameters_h
#define __deal2__cdr_1_parameters_h

#include <fstream>
#include <string>

#include <deal.II/base/parameter_handler.h>

using namespace dealii;
class Parameters
{
public:
  double inner_radius;
  double outer_radius;

  double diffusion_coefficient;
  std::string convection_field;
  double reaction_coefficient;

  unsigned int refinement_level;
  unsigned int fe_order;

  void read_parameter_file(std::string file_name);
private:
  void configure_parameter_handler(ParameterHandler &parameter_handler);
};
#endif
