#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

#include "../common/parameters.h"
#include "../common/write_pvtu_output.h"
#include "../common/assemble_system.h"

using namespace dealii;

constexpr int manifold_id {0};


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
  Vector<double> system_rhs;
  Vector<double> current_solution;
  SparseILU<double> preconditioner;

  void setup_geometry();
  void setup_system();
  void setup_dof_handler();
  void refine_mesh();
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
  triangulation.set_manifold(manifold_id, boundary_description);
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_id);
    }
  triangulation.refine_global(parameters.refinement_level);
  dof_handler.initialize(triangulation, fe);

  // This must be done here so that the vector is the correct size when
  // entering setup_system. During time iteration refine_mesh will resize
  // current_solution.
  current_solution.reinit(dof_handler.n_dofs());
}


// This should be called once the triangulation is set up (or has been
// refined). More specifically, this should be called after setup_geometry or
// refine_mesh.
template<int dim>
void CDRProblem<dim>::setup_dof_handler()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "number of DoFs: " << dof_handler.n_dofs() << std::endl;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, manifold_id, constraints);
  constraints.close();
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,
                                  constraints, /*keep_constrained_dofs*/true);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
}


template<int dim>
void CDRProblem<dim>::setup_system()
{
  system_rhs.reinit(dof_handler.n_dofs());
  system_matrix.reinit(sparsity_pattern);
  // This must be called with the template syntax (assemble_system<dim>, as
  // opposed to assemble_system) because one of the lambda output types
  // depends on dim.
  CDR::assemble_system<dim>
    (dof_handler, quad, convection_function, forcing_function, parameters,
     current_solution, constraints, system_matrix, system_rhs);
  preconditioner.initialize(system_matrix);
}


template<int dim>
void CDRProblem<dim>::time_iterate()
{
  double current_time = parameters.start_time;

  CDR::WritePVTUOutput pvtu_output(parameters.patch_level);

  for (unsigned int time_step_n = 0; time_step_n < parameters.n_time_steps;
       ++time_step_n)
    {
      current_time += time_step;

      SolverControl solver_control(current_solution.size(),
                                   1e-6*system_rhs.l2_norm(),
                                   /*log_history = */ false,
                                   /*log_result = */ false);
      SolverGMRES<Vector<double>> solver(solver_control);
      solver.solve(system_matrix, current_solution, system_rhs, preconditioner);
      constraints.distribute(current_solution);

      if (time_step_n % parameters.save_interval == 0)
        {
          pvtu_output.write_output(dof_handler, current_solution, time_step_n,
                                   current_time);
        }
      std::cout << time_step_n << std::endl;

      refine_mesh();
    }
}


// This function estimates the current solution error, refines (and coarsens)
// the mesh appropriately, and transfers the current solution onto the new mesh.
template<int dim>
void CDRProblem<dim>::refine_mesh()
{
  SolutionTransfer<dim> solution_transfer(dof_handler);

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate
    (dof_handler, QGauss<dim - 1>(fe.degree + 1), typename FunctionMap<dim>::type(),
     current_solution, estimated_error_per_cell);
  GridRefinement::refine(triangulation, estimated_error_per_cell, 1e-3);
  GridRefinement::coarsen(triangulation, estimated_error_per_cell, 5e-4);

  // TODO make max_refinement_level a parameter
  if (triangulation.n_levels() > parameters.refinement_level)
    {
      for (const auto &cell :
             triangulation.cell_iterators_on_level(parameters.refinement_level))
        {
          cell->clear_refine_flag();
        }
    }

  Vector<double> unrefined_current_solution {current_solution};

  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(unrefined_current_solution);

  triangulation.execute_coarsening_and_refinement();
  setup_dof_handler();
  current_solution.reinit(dof_handler.n_dofs());
  solution_transfer.interpolate(unrefined_current_solution, current_solution);

  constraints.distribute(current_solution);
  setup_system();
}


template<int dim>
void CDRProblem<dim>::run()
{
  setup_geometry();
  setup_dof_handler();
  setup_system();
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
    1.0, 2.0,
    1.0e-3, 1.0e-4, true,
    3, 2,
    0.0, 20.0, 2000,
    1, 3
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
