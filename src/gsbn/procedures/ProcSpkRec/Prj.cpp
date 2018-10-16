#include "gsbn/procedures/ProcSpkRec/Prj.hpp"

namespace gsbn
{
namespace proc_spk_rec
{

Prj::Prj(int &id, ProjParam prj_param, Database &db)
{
  _id = id;

  // DO NOT CHECK THE RETURN VALUE, SINCE THE SPIKE VECTOR MAYBE NOT IN THE CURRENT
  // RANK.
  CHECK(_zj2 = db.sync_vector_f32("zj2_" + to_string(_id)));
  CHECK(_zj2_nocol = db.sync_vector_f32("zj2nocol_" + to_string(_id)));
  CHECK(_ssi = db.sync_vector_i32(".ssi_" + to_string(_id)));
  CHECK(_ssj = db.sync_vector_i32(".ssj_" + to_string(_id)));

  id++;
}

Prj::~Prj()
{
}

void Prj::record_zj2(string filename, int simstep)
{
  const float *ptr_zj2 = _zj2->cpu_data();
  fstream output(filename, ios::out | ios::app);
  output << simstep << "," << ptr_zj2[99] << endl;
}
void Prj::record_zj2_nocol(string filename, int simstep)
{
  const float *ptr_zj2_nocol = _zj2_nocol->cpu_data();
  fstream output(filename, ios::out | ios::app);
  output << simstep << "," << ptr_zj2_nocol[99] << endl;
}
void Prj::record_ssi(string filename, int simstep)
{
  fstream output(filename, ios::out | ios::app);
  output << simstep << ",";
  bool active = false;
  for (int i = 0; i < _ssi->size(); i++)
  {
    if (_ssi->cpu_data()[i] == 9)
    {
      active = true;
      break;
    }
  }
  if (active)
  {
    output << 1 << endl;
  }
  else
  {
    output << 0 << endl;
  }
}
void Prj::record_ssj(string filename, int simstep)
{
  fstream output(filename, ios::out | ios::app);
  output << simstep << ",";
  bool active = false;
  for (int i = 0; i < _ssj->size(); i++)
  {
    if (_ssj->cpu_data()[i] == 9)
    {
      active = true;
      break;
    }
  }
  if (active)
  {
    output << 1 << endl;
  }
  else
  {
    output << 0 << endl;
  }
}

} // namespace proc_spk_rec
} // namespace gsbn
