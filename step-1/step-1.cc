#include <deal.II/base/quadrature_lib.h>

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

#include <array>
#include <functional>
#include <iostream>

#include "../common/parameters.h"
#include "../common/system_matrix.h"
#include "../common/system_rhs.h"
#include "../common/write_pvtu_output.h"
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

  const std::function<std::array<double, dim>(Point<dim>)> convection_function;
  const std::function<double(double, Point<dim>)> forcing_function;

  ConstraintMatrix constraints;
  SparseMatrix<double> system_matrix;
  SparseILU<double> preconditioner;

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
  convection_function
    {[](Point<dim> p) -> std::array<double, dim> {return {-p[1], p[0]};}},
  forcing_function
    {[](double t, Point<dim> p) -> double
        {return std::exp(-8*t)*std::exp(-40*Utilities::fixed_power<6>(p[0] - 1.5))
            *std::exp(-40*Utilities::fixed_power<6>(p[1]));}}
{
  Assert(dim == 2, ExcNotImplemented());
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
  CDR::create_system_matrix<dim>(dof_handler, quad, convection_function,
                                 parameters, time_step, constraints, system_matrix);
  preconditioner.initialize(system_matrix);
}


template<int dim>
void CDRProblem<dim>::time_iterate()
{
  Vector<double> current_solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  double current_time = parameters.start_time;
  CDR::WritePVTUOutput output(parameters.patch_level);
  for (unsigned int time_step_n = 0; time_step_n < parameters.n_time_steps;
       ++time_step_n)
    {
      current_time += time_step;
      system_rhs = 0.0;

      CDR::create_system_rhs<dim>
        (dof_handler, quad, convection_function, forcing_function, parameters,
         current_solution, constraints, current_time, system_rhs);

      SolverControl solver_control(dof_handler.n_dofs(),
                                   1e-6*system_rhs.l2_norm(),
                                   /*log_history = */ false,
                                   /*log_result = */ false);
      SolverGMRES<Vector<double>> solver(solver_control);
      solver.solve(system_matrix, current_solution, system_rhs, preconditioner);
      constraints.distribute(current_solution);

      if (time_step_n % parameters.save_interval == 0)
        {
          output.write_output(dof_handler, current_solution, time_step_n,
                              current_time);
        }
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
  auto t0 = std::chrono::high_resolution_clock::now();
  Utilities::MPI::MPI_InitFinalize mpi_initialization
    (argc, argv, numbers::invalid_unsigned_int);
  CDR::Parameters parameters
  {
    1.0, 2.0, // inner and outer radii
    1.0e-3, 1.0e-4, // diffusion and reaction coefficient
    true, // use time dependent forcing
    4, 2, // default refinement level, fe order
    0.0, 2, 2000, // start time, stop time, time steps
    1, 3 // save interval, patch level
  };
  CDRProblem<dim> cdr_problem(parameters);
  cdr_problem.run();

  auto t1 = std::chrono::high_resolution_clock::now();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "time elapsed: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                << " milliseconds."
                << std::endl;
    }

  return 0;
}
