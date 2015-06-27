#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/constraint_matrix.h>

namespace CDR
{
  using namespace dealii;

  template<int dim, typename Matrix>
  void create_convection_matrix(const DoFHandler<dim>     &dof_handler,
                                const QGauss<dim>         &quad,
                                const FunctionParser<dim> &convection_function,
                                Matrix                    &convection_matrix,
                                const ConstraintMatrix    &constraints);

  template<int dim, typename Matrix>
  void create_convection_matrix(const DoFHandler<dim>     &dof_handler,
                                const QGauss<dim>         &quad,
                                const FunctionParser<dim> &convection_function,
                                Matrix                    &convection_matrix);
}
