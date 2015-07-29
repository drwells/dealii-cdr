#ifndef dealii__cdr_parameters_h
#define dealii__cdr_parameters_h

#include <fstream>
#include <string>

#include <deal.II/base/parameter_handler.h>

namespace CDR
{
  using namespace dealii;

  class Parameters
  {
  public:
    double inner_radius;
    double outer_radius;

    double diffusion_coefficient;
    double reaction_coefficient;
    bool time_dependent_forcing;

    unsigned int refinement_level;
    unsigned int fe_order;

    double start_time;
    double stop_time;
    unsigned int n_time_steps;

    unsigned int save_interval;
    unsigned int patch_level;

    void read_parameter_file(std::string file_name);
  private:
    void configure_parameter_handler(ParameterHandler &parameter_handler);
  };
}
#endif
