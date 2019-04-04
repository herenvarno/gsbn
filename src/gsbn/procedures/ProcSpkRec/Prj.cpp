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
//  CHECK(_zi2 = db.sync_vector_f32("zi2_" + to_string(_id)));
//  CHECK(_zi2_nocol = db.sync_vector_f32("zi2nocol_" + to_string(_id)));
//  CHECK(_zj2 = db.sync_vector_f32("zj2_" + to_string(_id)));
//  CHECK(_zj2_nocol = db.sync_vector_f32("zj2nocol_" + to_string(_id)));
//  CHECK(_eij = db.sync_vector_f32("eij_" + to_string(_id)));
//  CHECK(_eij_nocol = db.sync_vector_f32("eijnocol_" + to_string(_id)));
//  CHECK(_pij = db.sync_vector_f32("pij_" + to_string(_id)));
//  CHECK(_pij_nocol = db.sync_vector_f32("pijnocol_" + to_string(_id)));
//  CHECK(_wij = db.sync_vector_f32("wij_" + to_string(_id)));
//  CHECK(_wij_nocol = db.sync_vector_f32("wijnocol_" + to_string(_id)));
  CHECK(_ssi = db.sync_vector_i32(".ssi_" + to_string(_id)));
  CHECK(_ssj = db.sync_vector_i32(".ssj_" + to_string(_id)));

  id++;
}

Prj::~Prj()
{
}

//void Prj::record_zi2(string filename, int simstep)
//{
//  const float *ptr_zi2 = _zi2->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_zi2[99] << endl;
//}
//void Prj::record_zi2_nocol(string filename, int simstep)
//{
//  const float *ptr_zi2_nocol = _zi2_nocol->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_zi2_nocol[99] << endl;
//}
//void Prj::record_zj2(string filename, int simstep)
//{
//  const float *ptr_zj2 = _zj2->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_zj2[99] << endl;
//}
//void Prj::record_zj2_nocol(string filename, int simstep)
//{
//  const float *ptr_zj2_nocol = _zj2_nocol->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_zj2_nocol[99] << endl;
//}
//void Prj::record_eij(string filename, int simstep)
//{
//  const float *ptr_eij = _eij->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_eij[99] << endl;
//}
//void Prj::record_eij_nocol(string filename, int simstep)
//{
//  const float *ptr_eij_nocol = _eij_nocol->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_eij_nocol[99] << endl;
//}
//void Prj::record_pij(string filename, int simstep)
//{
//  const float *ptr_pij = _pij->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_pij[99] << endl;
//}
//void Prj::record_pij_nocol(string filename, int simstep)
//{
//  const float *ptr_pij_nocol = _pij_nocol->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_pij_nocol[99] << endl;
//}
//void Prj::record_wij(string filename, int simstep)
//{
//  const float *ptr_wij = _wij->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_wij[99] << endl;
//}
//void Prj::record_wij_nocol(string filename, int simstep)
//{
//  const float *ptr_wij_nocol = _wij_nocol->cpu_data();
//  fstream output(filename, ios::out | ios::app);
//  output << simstep << "," << ptr_wij_nocol[99] << endl;
//}

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
