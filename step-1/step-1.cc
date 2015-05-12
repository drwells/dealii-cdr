#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "../common/convection.h"
#include "parameters.h"

using namespace dealii;

template<int dim>
class CDRProblem
{
public:
  CDRProblem(const Parameters &parameters);

private:
  const Parameters parameters;

  FE_Q<dim> fe;
  QGauss<dim> quad;
  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;

  FunctionParser<dim> convection_function;

  ConstraintMatrix constraints;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> convection_matrix;
  SparseMatrix<double> laplace_matrix;

  SparseMatrix<double> system_matrix;

  void setup_geometry();
  void setup_matrices();
};


template<int dim>
CDRProblem<dim>::CDRProblem(const Parameters &parameters) :
  parameters(parameters),
  fe(parameters.fe_order),
  quad(3*(2 + parameters.fe_order)/2)
{
  convection_function.initialize(std::string("x,y"), parameters.convection_field,
                                 std::map<std::string, double>());
}


template<int dim>
void CDRProblem<dim>::setup_geometry()
{
  const Point<dim> center(true);
  GridGenerator::hyper_shell(triangulation, center, parameters.inner_radius,
                             parameters.outer_radius);
  dof_handler.initialize(triangulation, fe);
}


template<int dim>
void CDRProblem<dim>::setup_matrices()
{
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
  {
    CompressedSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,
                                    constraints, /*keep_constrained_dofs*/false);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  }

  mass_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler, quad, mass_matrix, nullptr,
                                    constraints);
  convection_matrix.reinit(sparsity_pattern);
  create_convection_matrix(dof_handler, quad, convection_function,
                           convection_matrix, constraints);
  laplace_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_laplace_matrix(dof_handler, quad, laplace_matrix,
                                       nullptr, constraints);
  auto time_step = (parameters.stop_time - parameters.start_time)/
    parameters.n_time_steps;
  system_matrix.reinit(sparsity_pattern);
  system_matrix = 0.0;
  system_matrix.add(1.0, mass_matrix);
  system_matrix.add(time_step*parameters.diffusion_coefficient/2.0,
                    laplace_matrix);
  system_matrix.add(time_step/2.0, convection_matrix);
  system_matrix.add(time_step*parameters.reaction_coefficient/2.0, mass_matrix);
}


constexpr int dim {2};


int main(int argc, char *argv[])
{
  Parameters parameters
  {
    1.0, 2.0,
    1.0e-4, "exp(x)", 1.0,
    1, 2,
    0.0, 1.0, 100,
    10, 2
  };
  CDRProblem<dim> cdr_problem(parameters);

  return 0;
}
