#ifndef dealii__cdr_write_pvtu_output_h
#define dealii__cdr_write_pvtu_output_h
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <fstream>
#include <string>
#include <vector>


namespace CDR
{
  using namespace dealii;

  class WritePVTUOutput
  {
  public:
    WritePVTUOutput(const unsigned int patch_level);

    template<int dim, typename VectorType>
    void write_output(const DoFHandler<dim> &dof_handler,
                      const VectorType      &solution,
                      const unsigned int    &time_step_n,
                      const double          &current_time);
  private:
    const unsigned int patch_level;
    const unsigned int this_mpi_process;
  };
}
#endif
