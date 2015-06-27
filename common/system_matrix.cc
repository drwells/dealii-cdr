#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "parameters.h"
#include "system_matrix.h"
#include "system_matrix.templates.h"

namespace CDR
{
  using namespace dealii;

  template
  void create_system_matrix<2, SparseMatrix<double>>
    (const DoFHandler<2>     &dof_handler,
     const QGauss<2>         &quad,
     const FunctionParser<2> &convection_function,
     const ConstraintMatrix  &constraints,
     const CDR::Parameters   &parameters,
     SparseMatrix<double>    &system_matrix);

  template
  void create_system_matrix<3, SparseMatrix<double>>
    (const DoFHandler<3>     &dof_handler,
     const QGauss<3>         &quad,
     const FunctionParser<3> &convection_function,
     const ConstraintMatrix  &constraints,
     const CDR::Parameters   &parameters,
     SparseMatrix<double>    &system_matrix);

  template
  void create_system_matrix<2, TrilinosWrappers::SparseMatrix>
  (const DoFHandler<2>            &dof_handler,
   const QGauss<2>                &quad,
   const FunctionParser<2>        &convection_function,
   const ConstraintMatrix         &constraints,
   const CDR::Parameters          &parameters,
   TrilinosWrappers::SparseMatrix &system_matrix);

  template
  void create_system_matrix<3, TrilinosWrappers::SparseMatrix>
  (const DoFHandler<3>            &dof_handler,
   const QGauss<3>                &quad,
   const FunctionParser<3>        &convection_function,
   const ConstraintMatrix         &constraints,
   const CDR::Parameters          &parameters,
   TrilinosWrappers::SparseMatrix &system_matrix);

}
