#ifndef __GSBN_PROC_PLASTICITY_HPP
#define __GSBN_PROC_PLASTICITY_HPP

namespace gsbn{
namespace proc_plasticity

class ProcPlasticity : public ProcedureBase{

REGISTER(ProcPlasticity);

public:
	ProcPlasticity(){};
	~ProcPlasticity(){};
	
	void init_new(NetParam net_param, Database& db);
	void init_copy(NetParam net_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	vector<int> _avail_conn;
	vector<Proj*> _list_proj;
	vector<Pop*> _list_pop;
	vector<Hcu*> _list_hcu;
	vector<Conn*> _list_conn;
	
	SyncVector<int>* _spike;
};

}
}



#endif // __GSBN_PROC_PLASTICITY_HPP
