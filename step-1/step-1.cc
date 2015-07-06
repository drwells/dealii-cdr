#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "../common/parameters.h"
#include "../common/system_matrix.h"
#include "../common/system_rhs.h"
#include "../common/write_xdmf_output.h"

using namespace dealii;

template<int dim>
class CDRProblem
{
public:
  CDRProblem(const CDR::Parameters &parameters);
  void run();
private:
  const CDR::Parameters parameters;
  const double time_step;

  FE_Q<dim> fe;
  QGauss<dim> quad;
  const SphericalManifold<dim> boundary_description;
  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;

  std::map<std::string, double> parser_constants;
  FunctionParser<dim> convection_function;
  FunctionParser<dim> forcing_function;

  ConstraintMatrix constraints;

  SparseMatrix<double> system_matrix;

  SparseILU<double>    preconditioner;

  void setup_geometry();
  void setup_matrices();
  void time_iterate();
};


template<int dim>
CDRProblem<dim>::CDRProblem(const CDR::Parameters &parameters) :
  parameters(parameters),
  time_step {(parameters.stop_time - parameters.start_time)
    /parameters.n_time_steps},
  fe(parameters.fe_order),
  quad(3*(2 + parameters.fe_order)/2),
  boundary_description(Point<dim>(true)),
  convection_function(dim),
  forcing_function(1)
{
  Assert(dim == 2, ExcNotImplemented());
  parser_constants["pi"] = numbers::PI;
  std::vector<std::string> convection_field
  {
    parameters.convection_field.substr
      (0, parameters.convection_field.find_first_of(",")),
    parameters.convection_field.substr
      (parameters.convection_field.find_first_of(",") + 1)
  };

  convection_function.initialize(std::string("x,y"), convection_field,
                                 parser_constants,
                                 /*time_dependent=*/false);
  forcing_function.initialize(std::string("x,y,t"), parameters.forcing,
                              parser_constants,
                              /*time_dependent=*/true);
  forcing_function.set_time(parameters.start_time);
}


template<int dim>
void CDRProblem<dim>::setup_geometry()
{
  const Point<dim> center(true);
  GridGenerator::hyper_shell(triangulation, center, parameters.inner_radius,
                             parameters.outer_radius);
  triangulation.set_manifold(0, boundary_description);
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell->set_all_manifold_ids(0);
    }
  triangulation.refine_global(parameters.refinement_level);
  dof_handler.initialize(triangulation, fe);
  std::cout << "number of DoFs: " << dof_handler.n_dofs() << std::endl;
}


template<int dim>
void CDRProblem<dim>::setup_matrices()
{
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
  {
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,
                                    constraints, /*keep_constrained_dofs*/true);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  }

  system_matrix.reinit(sparsity_pattern);
  CDR::create_system_matrix(dof_handler, quad, convection_function, parameters,
                            time_step, constraints, system_matrix);
  preconditioner.initialize(system_matrix);
}


template<int dim>
void CDRProblem<dim>::time_iterate()
{
  Vector<double> current_solution(dof_handler.n_dofs());
  Vector<double> right_hand_side(dof_handler.n_dofs());

  double current_time = parameters.start_time;
  CDR::WriteXDMFOutput xdmf_output(parameters.patch_level,
                                   /*update_mesh_at_each_step*/false);
  for (unsigned int time_step_n = 0; time_step_n < parameters.n_time_steps;
       ++time_step_n)
    {
      current_time += time_step;
      forcing_function.advance_time(time_step);

      right_hand_side = 0.0;
      CDR::create_system_rhs(dof_handler, quad, convection_function,
                             forcing_function, parameters, current_solution,
                             constraints, right_hand_side);

      SolverControl solver_control(current_solution.size(),
                                   1e-6*right_hand_side.l2_norm());
      SolverGMRES<Vector<double>> solver(solver_control);
      solver.solve(system_matrix, current_solution, right_hand_side, preconditioner);
      constraints.distribute(current_solution);

      if (time_step_n % parameters.save_interval == 0)
        {
          xdmf_output.write_output(dof_handler, current_solution, time_step_n,
                                   current_time);
        }

      std::cout << time_step_n << std::endl;
    }
}


template<int dim>
void CDRProblem<dim>::run()
{
  setup_geometry();
  setup_matrices();
  time_iterate();
}


constexpr int dim {2};


int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization
    (argc, argv, numbers::invalid_unsigned_int);
  CDR::Parameters parameters
  {
    1.0, 2.0,
    1.0e-3, "-y,x", 1.0e-4, "exp(-2*t)*exp(-40*(x - 1.5)^6)"
    "*exp(-40*y^6)", true,
    3, 2,
    0.0, 20.0, 2000,
    1, 3
  };
  CDRProblem<dim> cdr_problem(parameters);
  cdr_problem.run();

  return 0;
}
