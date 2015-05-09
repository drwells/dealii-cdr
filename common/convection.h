using namespace dealii;
template<int dim>
void create_convection_matrix(const DoFHandler<dim> &dof_handler,
                              const QGauss<dim> &quad,
                              const FunctionParser<dim> &convection_function,
                              SparseMatrix<double> &convection_matrix);
