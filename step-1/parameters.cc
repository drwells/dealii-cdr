#include "parameters.h"

void Parameters::configure_parameter_handler(ParameterHandler &parameter_handler)
{
  parameter_handler.enter_subsection("Mesh Information");
  {
    parameter_handler.declare_entry
      ("inner_radius", "1.0", Patterns::Double(0.0), "Inner radius.");
    parameter_handler.declare_entry
      ("outer_radius", "2.0", Patterns::Double(0.0), "Outer radius.");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Physical Information");
  {
    parameter_handler.declare_entry
      ("diffusion_coefficient", "1.0", Patterns::Double(0.0), "Diffusion coefficient.");
    parameter_handler.declare_entry
      ("reaction_coefficient", "1.0", Patterns::Double(0.0), "Reaction coefficient.");
    parameter_handler.declare_entry
      ("expression_field", "(sin(pi/3.0), cos(pi/3.0))", Patterns::Anything(),
       "Convection field.");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Finite Element Information");
  {
    parameter_handler.declare_entry
      ("refinement_level", "1", Patterns::Integer(0), "Number of times to refine.");
    parameter_handler.declare_entry
      ("fe_order", "1", Patterns::Integer(1), "Finite element order.");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Time Step Information");
  {
    parameter_handler.declare_entry
      ("start_time", "0.0", Patterns::Double(0.0), "Start time.");
    parameter_handler.declare_entry
      ("stop_time", "1.0", Patterns::Double(1.0), "Stop time.");
    parameter_handler.declare_entry
      ("n_time_steps", "1.0", Patterns::Integer(1), "Number of time steps.");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Output Information");
  {
    parameter_handler.declare_entry
      ("save_interval", "10", Patterns::Integer(1), "Save interval.");
    parameter_handler.declare_entry
      ("patch_level", "2", Patterns::Integer(0), "Patch level.");
  }
  parameter_handler.leave_subsection();
}

void Parameters::read_parameter_file(std::string file_name)
{
  ParameterHandler parameter_handler;
  {
    std::ifstream file(file_name);
    configure_parameter_handler(parameter_handler);
    parameter_handler.read_input(file);
  }

  inner_radius = parameter_handler.get_double("inner_radius");
  outer_radius = parameter_handler.get_double("inner_radius");

  diffusion_coefficient = parameter_handler.get_double("diffusion_coefficient");
  reaction_coefficient = parameter_handler.get_double("reaction_coefficient");

  refinement_level = parameter_handler.get_integer("refinement_level");
  fe_order = parameter_handler.get_integer("fe_order");

  start_time = parameter_handler.get_double("start_time");
  stop_time = parameter_handler.get_double("stop_time");
  n_time_steps = parameter_handler.get_integer("n_time_steps");

  save_interval = parameter_handler.get_integer("save_interval");
  patch_level = parameter_handler.get_integer("patch_level");

}
