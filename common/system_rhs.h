#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/constraint_matrix.h>

#include "parameters.h"

namespace CDR
{
  using namespace dealii;

  template<int dim, typename VectorType>
  void create_system_rhs(const DoFHandler<dim>     &dof_handler,
                         const QGauss<dim>         &quad,
                         const FunctionParser<dim> &convection_function,
                         FunctionParser<dim>       &forcing_function,
                         const CDR::Parameters     &parameters,
                         const VectorType          &previous_solution,
                         const ConstraintMatrix    &constraints,
                         VectorType                &system_rhs);
}
