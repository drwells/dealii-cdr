using namespace dealii;
template<int dim>
void create_convection_matrix(const DoFHandler<dim>     &dof_handler,
                              const QGauss<dim>         &quad,
                              const FunctionParser<dim> &convection_function,
                              SparseMatrix<double>      &convection_matrix,
                              const ConstraintMatrix    &constraints)
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
                  cell_matrix(i, j) += fe_values.shape_value(i, q)
                    *convection_contribution
                    *fe_values.JxW(q);
                }
            }
        }
      constraints.distribute_local_to_global(cell_matrix, local_indices,
                                             convection_matrix);
    }
}
