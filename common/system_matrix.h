#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/constraint_matrix.h>

using namespace dealii;
template<int dim, typename Matrix>
void create_system_matrix(const DoFHandler<dim>     &dof_handler,
                          const QGauss<dim>         &quad,
                          const FunctionParser<dim> &convection_function,
                          const ConstraintMatrix    &constraints,
                          const CDR::Parameters     &parameters,
                          Matrix                    &system_matrix);
