#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "system_rhs.templates.h"

namespace CDR
{
  using namespace dealii;

  template
  void create_system_rhs<2, Vector<double>>
  (const DoFHandler<2>     &dof_handler,
   const QGauss<2>         &quad,
   const FunctionParser<2> &convection_function,
   FunctionParser<2>       &forcing_function,
   const CDR::Parameters   &parameters,
   const Vector<double>    &previous_solution,
   const ConstraintMatrix  &constraints,
   Vector<double>          &system_rhs);

  template
  void create_system_rhs<3, Vector<double>>
  (const DoFHandler<3>     &dof_handler,
   const QGauss<3>         &quad,
   const FunctionParser<3> &convection_function,
   FunctionParser<3>       &forcing_function,
   const CDR::Parameters   &parameters,
   const Vector<double>    &previous_solution,
   const ConstraintMatrix  &constraints,
   Vector<double>          &system_rhs);

  template
  void create_system_rhs<2, TrilinosWrappers::MPI::Vector>
  (const DoFHandler<2>                 &dof_handler,
   const QGauss<2>                     &quad,
   const FunctionParser<2>             &convection_function,
   FunctionParser<2>                   &forcing_function,
   const CDR::Parameters               &parameters,
   const TrilinosWrappers::MPI::Vector &previous_solution,
   const ConstraintMatrix              &constraints,
   TrilinosWrappers::MPI::Vector       &system_rhs);

  template
  void create_system_rhs<3, TrilinosWrappers::MPI::Vector>
  (const DoFHandler<3>                 &dof_handler,
   const QGauss<3>                     &quad,
   const FunctionParser<3>             &convection_function,
   FunctionParser<3>                   &forcing_function,
   const CDR::Parameters               &parameters,
   const TrilinosWrappers::MPI::Vector &previous_solution,
   const ConstraintMatrix              &constraints,
   TrilinosWrappers::MPI::Vector       &system_rhs);
}
