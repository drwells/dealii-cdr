#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "../common/convection.h"
#include "parameters.h"

using namespace dealii;

constexpr int manifold_id {0};


template<int dim>
class CDRProblem
{
public:
  CDRProblem(const Parameters &parameters);
  void run();
private:
  const Parameters parameters;
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

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> convection_matrix;
  SparseMatrix<double> laplace_matrix;

  SparseMatrix<double> system_matrix;
  SparseMatrix<double> right_hand_side_matrix;

  SparseILU<double>    preconditioner;

  Vector<double> right_hand_side;
  Vector<double> current_forcing;
  Vector<double> previous_forcing;
  Vector<double> current_solution;
  Vector<double> previous_solution;

  void setup_geometry();
  void setup_matrices();
  void refine_mesh();
  void time_iterate();
};


template<int dim>
CDRProblem<dim>::CDRProblem(const Parameters &parameters) :
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
  std::vector<std::string> convection_field(2);
  convection_field[0] =
    parameters.convection_field.substr(0, parameters.convection_field.find_first_of(","));
  convection_field[1] =
    parameters.convection_field.substr(parameters.convection_field.find_first_of(",") + 1);

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
  triangulation.set_manifold(manifold_id, boundary_description);
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_id);
    }
  triangulation.refine_global(parameters.refinement_level);
  dof_handler.initialize(triangulation, fe);

  right_hand_side.reinit(dof_handler.n_dofs());
  current_forcing.reinit(dof_handler.n_dofs());
  previous_forcing.reinit(dof_handler.n_dofs());
  current_solution.reinit(dof_handler.n_dofs());
  previous_solution.reinit(dof_handler.n_dofs());
}


template<int dim>
void CDRProblem<dim>::setup_matrices()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "number of DoFs: " << dof_handler.n_dofs() << std::endl;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, manifold_id, constraints);
  constraints.close();
  {
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,
                                    constraints, /*keep_constrained_dofs*/true);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  }

  mass_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler, quad, mass_matrix);
  convection_matrix.reinit(sparsity_pattern);
  create_convection_matrix(dof_handler, quad, convection_function,
                           convection_matrix);
  laplace_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_laplace_matrix(dof_handler, quad, laplace_matrix);

  {
    system_matrix.reinit(sparsity_pattern);
    create_system_matrix(dof_handler, quad, convection_function, constraints,
                         parameters, system_matrix);

    preconditioner.initialize(system_matrix);
  }

  {
    right_hand_side_matrix.reinit(sparsity_pattern);
    right_hand_side_matrix = 0.0;
    right_hand_side_matrix.add
      (1.0 - time_step*parameters.reaction_coefficient/2.0, mass_matrix);
    right_hand_side_matrix.add
      (-1.0*time_step*parameters.diffusion_coefficient/2.0, laplace_matrix);
    right_hand_side_matrix.add(-1.0*time_step/2.0, convection_matrix);
  }
}


template<int dim>
void CDRProblem<dim>::time_iterate()
{
  std::vector<XDMFEntry> xdmf_entries;
  std::string xdmf_file_name {"solution.xdmf"};


  double current_time = parameters.start_time;
  for (unsigned int time_step_n = 0; time_step_n < parameters.n_time_steps;
       ++time_step_n)
    {
      current_time += time_step;
      VectorTools::create_right_hand_side
        (dof_handler, quad, forcing_function, previous_forcing);
      if (parameters.time_dependent_forcing)
        {
          forcing_function.advance_time(time_step);
          VectorTools::create_right_hand_side
            (dof_handler, quad, forcing_function, current_forcing);
        }
      else
        {
          current_forcing = previous_forcing;
        }
      right_hand_side = 0.0;
      right_hand_side.add(time_step/2.0, current_forcing);
      right_hand_side.add(time_step/2.0, previous_forcing);
      right_hand_side_matrix.vmult_add(right_hand_side, previous_solution);

      constraints.condense(system_matrix);

      SolverControl solver_control(current_solution.size(),
                                   1e-6*(right_hand_side.l2_norm()));
      SolverGMRES<Vector<double>> solver(solver_control);
      solver.solve(system_matrix, current_solution, right_hand_side, preconditioner);

      if (time_step_n % parameters.save_interval == 0)
        {
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation;
          data_component_interpretation
            .push_back(DataComponentInterpretation::component_is_scalar);

          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(current_solution, "u",
                                   DataOut<dim>::type_dof_data,
                                   data_component_interpretation);
          data_out.build_patches(parameters.patch_level);

          std::string solution_file_name = "solution-" +
            Utilities::int_to_string(time_step_n, 9) + ".h5";
          std::string mesh_file_name = "mesh-" +
            Utilities::int_to_string(time_step_n, 9) + ".h5";
          DataOutBase::DataOutFilter data_filter
            (DataOutBase::DataOutFilterFlags(true, true));
          data_out.write_filtered_data(data_filter);
          data_out.write_hdf5_parallel
            (data_filter, /*write_mesh=*/true, mesh_file_name,
             solution_file_name, MPI_COMM_WORLD);
          data_out.write_xdmf_file(xdmf_entries, xdmf_file_name, MPI_COMM_WORLD);
          auto new_xdmf_entry = data_out.create_xdmf_entry
            (data_filter, mesh_file_name, solution_file_name,
             current_time, MPI_COMM_WORLD);
          xdmf_entries.push_back(std::move(new_xdmf_entry));
        }

      if (parameters.time_dependent_forcing)
        {
          std::swap(current_forcing, previous_forcing);
        }
      std::swap(current_solution, previous_solution);
      std::cout << time_step_n << std::endl;

      refine_mesh();
    }
}


template<int dim>
void CDRProblem<dim>::refine_mesh()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate
    (dof_handler, QGauss<dim - 1>(fe.degree + 1), typename FunctionMap<dim>::type(),
     previous_solution, estimated_error_per_cell);
  // GridRefinement::refine_and_coarsen_fixed_fraction
  //   (triangulation, estimated_error_per_cell, 0.2, 0.1);
  GridRefinement::refine(triangulation, estimated_error_per_cell, 1e-3);
  GridRefinement::coarsen(triangulation, estimated_error_per_cell, 1e-6);

  // TODO make max_refinement_level a parameter
  if (triangulation.n_levels() > parameters.refinement_level)
    {
      for (const auto &cell :
             triangulation.cell_iterators_on_level(parameters.refinement_level))
        {
          cell->clear_refine_flag();
        }
    }

  Vector<double> unrefined_previous_solution = previous_solution;

  SolutionTransfer<dim> solution_transfer(dof_handler);
  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(unrefined_previous_solution);

  triangulation.execute_coarsening_and_refinement();
  setup_matrices();

  right_hand_side.reinit(dof_handler.n_dofs());
  previous_forcing.reinit(dof_handler.n_dofs());
  current_forcing.reinit(dof_handler.n_dofs());
  previous_solution.reinit(dof_handler.n_dofs());
  current_solution.reinit(dof_handler.n_dofs());

  solution_transfer.interpolate(unrefined_previous_solution, previous_solution);
  constraints.distribute(previous_solution);
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
  Parameters parameters
  {
    1.0, 2.0,
    // 1.0e-2, "cos(pi/3),sin(pi/3)", 1.0, "1.0", false,
    1.0e-6, "-y,x", 1.0e-4, "exp(-10*(x - 1.5*cos(0.6*t))^2)"
    "*exp(-10*(y - 1.5*sin(0.6*t))^2)", true,
    3, 2,
    0.0, 20.0, 4000,
    1, 3
  };
  CDRProblem<dim> cdr_problem(parameters);
  cdr_problem.run();

  return 0;
}
