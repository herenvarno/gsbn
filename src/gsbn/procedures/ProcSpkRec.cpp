#include "gsbn/procedures/ProcSpkRec.hpp"

namespace gsbn
{
namespace proc_spk_rec
{

REGISTERIMPL(ProcSpkRec);

void ProcSpkRec::init_new(SolverParam solver_param, Database &db)
{
	_db = &db;

	string log_dir;
	CHECK(_glv.gets("log-dir", log_dir));
	CHECK(!log_dir.empty());

	string dir = log_dir + __PROC_NAME__;

	struct stat info;
	/* Check directory exists */
	if (stat(dir.c_str(), &info) == 0 && (info.st_mode & S_IFDIR))
	{
		_directory = dir;
	}
	else
	{
		LOG(WARNING) << "Directory does not exist! Create one!";
		string cmd = "mkdir -p " + dir;
		if (system(cmd.c_str()) != 0)
		{
			LOG(FATAL) << "Cannot create directory for state records! Aboart!";
		}
		_directory = dir;
	}

	ProcParam proc_param = get_proc_param(solver_param);

	Parser par(proc_param);
	if (!par.argi("offset", _offset))
	{
		_offset = 0;
	}
	else
	{
		CHECK_GE(_offset, 0);
	}
	if (!par.argi("period", _period))
	{
		_period = 1;
	}
	else
	{
		CHECK_GT(_period, 0);
	}

	CHECK(_glv.geti("rank", _rank));
	float dt;
	CHECK(_glv.getf("dt", dt));

	NetParam net_param = solver_param.net_param();

	int pop_id = 0;
	int pop_param_size = net_param.pop_param_size();
	for (int i = 0; i < pop_param_size; i++)
	{
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for (int j = 0; j < pop_num; j++)
		{
			Pop p(pop_id, pop_param, db);
			if (p._rank == _rank)
			{
				_pop_list.push_back(p);
			}
		}
	}

	int prj_id = 0;
	int prj_param_size = net_param.proj_param_size();
	for (int i = 0; i < prj_param_size; i++)
	{
		ProjParam prj_param = net_param.proj_param(i);
		Prj p(prj_id, prj_param, db);
		//		if (p._rank == _rank)
		//		{
		_prj_list.push_back(p);
		//		}
	}

	for (int i = 0; i < _pop_list.size(); i++)
	{
		Pop p = _pop_list[i];
		string filename = _directory + "/spk_pop_" + to_string(p._id) + ".csv";
		fstream output(filename, ios::out | std::ofstream::trunc);
		output << p._dim_hcu << "," << p._dim_mcu << "," << dt << "," << p._maxfq << endl;
		output.close();
	}

	for (int i = 0; i < _prj_list.size(); i++)
	{
		Prj p = _prj_list[i];
		string filename;
//		filename = _directory + "/zi2_pop_" + to_string(p._id) + ".csv";
//		fstream output1(filename, ios::out | std::ofstream::trunc);
//		output1.close();
//		filename = _directory + "/zi2_nocol_pop_" + to_string(p._id) + ".csv";
//		fstream output2(filename, ios::out | std::ofstream::trunc);
//		output2.close();
//		filename = _directory + "/zj2_pop_" + to_string(p._id) + ".csv";
//		fstream output3(filename, ios::out | std::ofstream::trunc);
//		output3.close();
//		filename = _directory + "/zj2_nocol_pop_" + to_string(p._id) + ".csv";
//		fstream output4(filename, ios::out | std::ofstream::trunc);
//		output4.close();
//		filename = _directory + "/eij_pop_" + to_string(p._id) + ".csv";
//		fstream output5(filename, ios::out | std::ofstream::trunc);
//		output5.close();
//		filename = _directory + "/eij_nocol_pop_" + to_string(p._id) + ".csv";
//		fstream output6(filename, ios::out | std::ofstream::trunc);
//		output6.close();
//		filename = _directory + "/pij_pop_" + to_string(p._id) + ".csv";
//		fstream output7(filename, ios::out | std::ofstream::trunc);
//		output7.close();
//		filename = _directory + "/pij_nocol_pop_" + to_string(p._id) + ".csv";
//		fstream output8(filename, ios::out | std::ofstream::trunc);
//		output8.close();
//		filename = _directory + "/wij_pop_" + to_string(p._id) + ".csv";
//		fstream output9(filename, ios::out | std::ofstream::trunc);
//		output9.close();
//		filename = _directory + "/wij_nocol_pop_" + to_string(p._id) + ".csv";
//		fstream output10(filename, ios::out | std::ofstream::trunc);
//		output10.close();
		filename = _directory + "/ssi_pop_" + to_string(p._id) + ".csv";
		fstream output11(filename, ios::out | std::ofstream::trunc);
		output11.close();
		filename = _directory + "/ssj_pop_" + to_string(p._id) + ".csv";
		fstream output12(filename, ios::out | std::ofstream::trunc);
		output12.close();
	}
}

void ProcSpkRec::init_copy(SolverParam solver_param, Database &db)
{
	init_new(solver_param, db);
}

void ProcSpkRec::update_cpu()
{

	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if (cycle_flag != 1)
	{
		return;
	}

	int simstep;
	float dt;
	CHECK(_glv.geti("simstep", simstep));
	CHECK_GE(simstep, 0);

	if (simstep < _offset)
	{
		return;
	}

	if ((simstep % _period) == 0)
	{
		for (int i = 0; i < _pop_list.size(); i++)
		{
			Pop p = _pop_list[i];
			string filename = _directory + "/spk_pop_" + to_string(p._id) + ".csv";
			p.record(filename, simstep);
		}
		for (int i = 0; i < _prj_list.size(); i++)
		{
			Prj p = _prj_list[i];
			string filename;
//			filename = _directory + "/zi2_pop_" + to_string(p._id) + ".csv";
//			p.record_zi2(filename, simstep);
//			filename = _directory + "/zi2_nocol_pop_" + to_string(p._id) + ".csv";
//			p.record_zi2_nocol(filename, simstep);
//			filename = _directory + "/zj2_pop_" + to_string(p._id) + ".csv";
//			p.record_zj2(filename, simstep);
//			filename = _directory + "/zj2_nocol_pop_" + to_string(p._id) + ".csv";
//			p.record_zj2_nocol(filename, simstep);
//			filename = _directory + "/eij_pop_" + to_string(p._id) + ".csv";
//			p.record_eij(filename, simstep);
//			filename = _directory + "/eij_nocol_pop_" + to_string(p._id) + ".csv";
//			p.record_eij_nocol(filename, simstep);
//			filename = _directory + "/pij_pop_" + to_string(p._id) + ".csv";
//			p.record_pij(filename, simstep);
//			filename = _directory + "/pij_nocol_pop_" + to_string(p._id) + ".csv";
//			p.record_pij_nocol(filename, simstep);
//			filename = _directory + "/wij_pop_" + to_string(p._id) + ".csv";
//			p.record_wij(filename, simstep);
//			filename = _directory + "/wij_nocol_pop_" + to_string(p._id) + ".csv";
//			p.record_wij_nocol(filename, simstep);
			filename = _directory + "/ssj_pop_" + to_string(p._id) + ".csv";
			p.record_ssj(filename, simstep);
			filename = _directory + "/ssi_pop_" + to_string(p._id) + ".csv";
			p.record_ssi(filename, simstep);
		}
	}
}

#ifndef CPU_ONLY
void ProcSpkRec::update_gpu()
{
	update_cpu();
}
#endif
} // namespace proc_spk_rec
} // namespace gsbn
