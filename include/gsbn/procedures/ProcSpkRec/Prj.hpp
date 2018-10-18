#ifndef __GSBN_PROC_SPK_REC_PRJ_HPP__
#define __GSBN_PROC_SPK_REC_PRJ_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Parser.hpp"
#include "gsbn/GlobalVar.hpp"

namespace gsbn
{
namespace proc_spk_rec
{

class Prj
{

public:
  Prj(int &id, ProjParam prj_param, Database &db);
  ~Prj();

  void record_zi2(string filename, int simstep);
  void record_zi2_nocol(string filename, int simstep);
  void record_zj2(string filename, int simstep);
  void record_zj2_nocol(string filename, int simstep);
  void record_eij(string filename, int simstep);
  void record_eij_nocol(string filename, int simstep);
  void record_pij(string filename, int simstep);
  void record_pij_nocol(string filename, int simstep);
  void record_wij(string filename, int simstep);
  void record_wij_nocol(string filename, int simstep);
  void record_ssj(string filename, int simstep);
  void record_ssi(string filename, int simstep);

  int _rank;
  int _id;

  SyncVector<float> *_zi2;
  SyncVector<float> *_zi2_nocol;
  SyncVector<float> *_zj2;
  SyncVector<float> *_zj2_nocol;
  SyncVector<float> *_eij;
  SyncVector<float> *_eij_nocol;
  SyncVector<float> *_pij;
  SyncVector<float> *_pij_nocol;
  SyncVector<float> *_wij;
  SyncVector<float> *_wij_nocol;
  SyncVector<int> *_ssi;
  SyncVector<int> *_ssj;
};

} // namespace proc_spk_rec

} // namespace gsbn

#endif //__GSBN_PROC_SPK_REC_POP_HPP__
