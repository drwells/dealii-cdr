#ifndef dealii__cdr_assemble_system_h
#define dealii__cdr_assemble_system_h
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/constraint_matrix.h>

#include <array>
#include <functional>

#include "parameters.h"

namespace CDR
{
  using namespace dealii;

  template<int dim, typename MatrixType, typename VectorType>
  void assemble_system
  (const DoFHandler<dim>                                    &dof_handler,
   const QGauss<dim>                                        &quad,
   const std::function<std::array<double, dim>(Point<dim>)> &convection_function,
   const std::function<double(double, Point<dim>)>          &forcing_function,
   const CDR::Parameters                                    &parameters,
   const VectorType                                         &current_solution,
   const ConstraintMatrix                                   &constraints,
   MatrixType                                               &system_matrix,
   VectorType                                               &system_rhs);
}
#endif
