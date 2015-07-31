#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/lac/sparse_matrix.h>

#include "assemble_system.templates.h"

namespace CDR
{
  using namespace dealii;

  template
  void assemble_system<2, SparseMatrix<double>, Vector<double>>
    (const DoFHandler<2>                                  &dof_handler,
     const QGauss<2>                                      &quad,
     const std::function<std::array<double, 2>(Point<2>)> &convection_function,
     const std::function<double(double, Point<2>)>        &forcing_function,
     const CDR::Parameters                                &parameters,
     const Vector<double>                                 &current_solution,
     const ConstraintMatrix                               &constraints,
     const double                                         current_time,
     SparseMatrix<double>                                 &system_matrix,
     Vector<double>                                       &system_rhs);

  template
  void assemble_system<3, SparseMatrix<double>, Vector<double>>
    (const DoFHandler<3>                                  &dof_handler,
     const QGauss<3>                                      &quad,
     const std::function<std::array<double, 3>(Point<3>)> &convection_function,
     const std::function<double(double, Point<3>)>        &forcing_function,
     const CDR::Parameters                                &parameters,
     const Vector<double>                                 &current_solution,
     const ConstraintMatrix                               &constraints,
     const double                                         current_time,
     SparseMatrix<double>                                 &system_matrix,
     Vector<double>                                       &system_rhs);

  template
  void assemble_system<2, TrilinosWrappers::SparseMatrix,
                       TrilinosWrappers::MPI::Vector>
  (const DoFHandler<2>                                  &dof_handler,
   const QGauss<2>                                      &quad,
   const std::function<std::array<double, 2>(Point<2>)> &convection_function,
   const std::function<double(double, Point<2>)>        &forcing_function,
   const CDR::Parameters                                &parameters,
   const TrilinosWrappers::MPI::Vector                  &current_solution,
   const ConstraintMatrix                               &constraints,
   const double                                         current_time,
   TrilinosWrappers::SparseMatrix                       &system_matrix,
   TrilinosWrappers::MPI::Vector                        &system_rhs);

  template
  void assemble_system<3, TrilinosWrappers::SparseMatrix,
                       TrilinosWrappers::MPI::Vector>
  (const DoFHandler<3>                                  &dof_handler,
   const QGauss<3>                                      &quad,
   const std::function<std::array<double, 3>(Point<3>)> &convection_function,
   const std::function<double(double, Point<3>)>        &forcing_function,
   const CDR::Parameters                                &parameters,
   const TrilinosWrappers::MPI::Vector                  &current_solution,
   const ConstraintMatrix                               &constraints,
   const double                                         current_time,
   TrilinosWrappers::SparseMatrix                       &system_matrix,
   TrilinosWrappers::MPI::Vector                        &system_rhs);
}
