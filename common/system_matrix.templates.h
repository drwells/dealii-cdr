#ifndef dealii__cdr_system_matrix_templates_h
#define dealii__cdr_system_matrix_templates_h
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include "system_matrix.h"

#include <vector>

namespace CDR
{
  using namespace dealii;

  template<int dim, typename UpdateFunction>
  void internal_create_system_matrix
  (const DoFHandler<dim>                                    &dof_handler,
   const QGauss<dim>                                        &quad,
   const std::function<std::array<double, dim>(Point<dim>)> &convection_function,
   const CDR::Parameters                                    &parameters,
   const double                                             &time_step,
   UpdateFunction                                           update_system_matrix)
  {
    auto &fe = dof_handler.get_fe();
    const auto dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FEValues<dim> fe_values(fe, quad, update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> local_indices(dofs_per_cell);
    Vector<double> current_convection(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell_matrix = 0.0;
            cell->get_dof_indices(local_indices);
            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                const auto current_convection
                {convection_function(fe_values.quadrature_point(q))};

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
            update_system_matrix(local_indices, cell_matrix);
          }
      }
  }

  template<int dim, typename Matrix>
  void create_system_matrix
  (const DoFHandler<dim>                                    &dof_handler,
   const QGauss<dim>                                        &quad,
   const std::function<std::array<double, dim>(Point<dim>)> &convection_function,
   const CDR::Parameters                                    &parameters,
   const double                                             &time_step,
   const ConstraintMatrix                                   &constraints,
   Matrix                                                   &system_matrix)
  {
    internal_create_system_matrix<dim>
      (dof_handler, quad, convection_function, parameters, time_step,
       [&constraints, &system_matrix](auto &local_indices, auto &cell_matrix)
       {
         constraints.distribute_local_to_global
           (cell_matrix, local_indices, system_matrix);
       });
  }

  template<int dim, typename Matrix>
  void create_system_matrix
  (const DoFHandler<dim>                                    &dof_handler,
   const QGauss<dim>                                        &quad,
   const std::function<std::array<double, dim>(Point<dim>)> &convection_function,
   const CDR::Parameters                                    &parameters,
   const double                                             &time_step,
   Matrix                                                   &system_matrix)
  {
    internal_create_system_matrix<dim>
      (dof_handler, quad, convection_function, parameters, time_step,
       [&system_matrix](auto &local_indices, auto &cell_matrix)
       {
         system_matrix.add(local_indices, cell_matrix);
       });
  }
}
#endif
