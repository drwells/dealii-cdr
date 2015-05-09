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

  parameter_handler.enter_subsection("Finite Element Information");
  {
    parameter_handler.declare_entry
      ("refinement_level", "1", Patterns::Integer(0), "Number of times to refine.");
    parameter_handler.declare_entry
      ("fe_order", "1", Patterns::Integer(1), "Finite element order.");
  }
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
}
