#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/constraint_matrix.h>

#include "parameters.h"

#include <vector>

namespace CDR
{
  using namespace dealii;

  template<int dim, typename Matrix>
  void create_system_matrix(const DoFHandler<dim>     &dof_handler,
                            const QGauss<dim>         &quad,
                            const FunctionParser<dim> &convection_function,
                            const ConstraintMatrix    &constraints,
                            const CDR::Parameters     &parameters,
                            Matrix                    &system_matrix)
  {
    auto &fe = dof_handler.get_fe();
    const auto dofs_per_cell = fe.dofs_per_cell;
    const double time_step = (parameters.stop_time - parameters.start_time)
      /parameters.n_time_steps;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FEValues<dim> fe_values(fe, quad, update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> local_indices(dofs_per_cell);
    Vector<double> current_convection(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrix = 0.0;
        cell->get_dof_indices(local_indices);
        for (unsigned int q = 0; q < quad.size(); ++q)
          {
            convection_function.vector_value(fe_values.quadrature_point(q),
                                             current_convection);
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
                    cell_matrix(i, j) += fe_values.JxW(q)*
                      // mass and reaction part
                      ((1.0 + time_step/2.0*parameters.reaction_coefficient)
                       *fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                       + time_step/2.0*
                       // convection part
                       (fe_values.shape_value(i, q)*convection_contribution
                        // Laplacian part
                        + parameters.diffusion_coefficient
                        *(fe_values.shape_grad(i, q)*fe_values.shape_grad(j, q)))
                       );
                  }
              }
          }
        constraints.distribute_local_to_global(cell_matrix, local_indices,
                                               system_matrix);
      }
  }
}
