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

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> convection_matrix;
  SparseMatrix<double> laplace_matrix;
};


template<int dim>
CDRProblem<dim>::CDRProblem(const Parameters &parameters) :
  parameters (parameters),
  fe (parameters.fe_order),
  quad (3*(2 + parameters.fe_order)/2)
{
  const Point<dim> center (true);
  GridGenerator::hyper_shell (triangulation, center, parameters.inner_radius,
                              parameters.outer_radius);
  dof_handler.initialize(triangulation, fe);
  {
    CompressedSparsityPattern dynamic_sparsity_pattern (dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dynamic_sparsity_pattern);
    sparsity_pattern.copy_from (dynamic_sparsity_pattern);
  }

  convection_function.initialize(std::string("x,y"), parameters.expression,
                                 std::map<std::string, double>());

  mass_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix (dof_handler, quad, mass_matrix);
  convection_matrix.reinit (sparsity_pattern);
  create_convection_matrix (dof_handler, quad, convection_function,
                            convection_matrix);
  laplace_matrix.reinit (sparsity_pattern);
  MatrixCreator::create_laplace_matrix (dof_handler, quad, laplace_matrix);
}


constexpr int dim = 2;


int main(int argc, char *argv[])
{
  Parameters parameters {1, "exp(x)", 1, 2};
  CDRProblem<dim> cdr_problem(parameters);

  return 0;
}
