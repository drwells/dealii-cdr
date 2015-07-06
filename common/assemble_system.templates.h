#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/vector.h>

#include <vector>

#include "assemble_system.h"

namespace CDR
{
  using namespace dealii;

  template<int dim, typename MatrixType, typename VectorType>
  void assemble_system(const DoFHandler<dim>     &dof_handler,
                       const QGauss<dim>         &quad,
                       const FunctionParser<dim> &convection_function,
                       FunctionParser<dim>       &forcing_function,
                       const CDR::Parameters     &parameters,
                       const VectorType          &current_solution,
                       const ConstraintMatrix    &constraints,
                       MatrixType                &system_matrix,
                       VectorType                &system_rhs)
  {
    auto &fe = dof_handler.get_fe();
    const auto dofs_per_cell = fe.dofs_per_cell;
    const double time_step = (parameters.stop_time - parameters.start_time)
      /parameters.n_time_steps;
    FEValues<dim> fe_values(fe, quad, update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    Vector<double> current_fe_coefficients(dofs_per_cell);
    std::vector<types::global_dof_index> local_indices(dofs_per_cell);

    Vector<double> current_convection(dim);

    const double current_time {forcing_function.get_time()};
    const double previous_time {current_time - time_step};

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell_rhs = 0.0;
            cell_matrix = 0.0;
            cell->get_dof_indices(local_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                current_fe_coefficients[i] = current_solution[local_indices[i]];
              }

            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                convection_function.vector_value(fe_values.quadrature_point(q),
                                                 current_convection);

                forcing_function.set_time(current_time);
                const double current_forcing = forcing_function.value
                  (fe_values.quadrature_point(q));
                forcing_function.set_time(previous_time);
                const double previous_forcing = forcing_function.value
                  (fe_values.quadrature_point(q));
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        double convection_contribution = 0.0;
                        for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
                          {
                            convection_contribution +=
                              current_convection[dim_n]
                              * fe_values.shape_grad(j, q)[dim_n];
                          }

                        const double mass_part
                        {fe_values.shape_value(i, q)*fe_values.shape_value(j, q)};
                        const double system_part
                        {time_step/2.0*
                            // convection part
                            (fe_values.shape_value(i, q)*convection_contribution
                             // diffusion part
                             + parameters.diffusion_coefficient
                             *fe_values.shape_grad(i, q)*fe_values.shape_grad(j, q)
                             // reaction part
                             + parameters.reaction_coefficient*mass_part)};
                        cell_rhs(i) += fe_values.JxW(q)
                          *((mass_part - system_part)*current_fe_coefficients[j]
                            + time_step/2.0*fe_values.shape_value(i, q)
                            *(current_forcing + previous_forcing));
                        cell_matrix(i, j) +=
                          fe_values.JxW(q)*(mass_part + system_part);
                      }
                  }
              }
            constraints.distribute_local_to_global
              (cell_matrix, cell_rhs, local_indices, system_matrix, system_rhs, true);
          }
        forcing_function.set_time(current_time);
      }
  }
}
