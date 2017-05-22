#ifndef __GSBN_PROCEDURE_FACTORY_HPP__
#define __GSBN_PROCEDURE_FACTORY_HPP__

#include "gsbn/GlobalVar.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Parser.hpp"

namespace gsbn{


/**
 * \class ProcedureBase
 * \brief ProcedureBase class is a template class to create new procedures.
 * Procedure in GSBN is a kind of plug-in module. The interface should be designed
 * based on the ProcedureBase
 */
class ProcedureBase{

public:
	ProcedureBase(){};
	virtual ~ProcedureBase(){};
	
	virtual void init_new(SolverParam net_param, Database& db) = 0;
	virtual void init_copy(SolverParam net_param, Database& db) = 0;
	virtual void update_cpu() = 0;
	#ifndef CPU_ONLY
	virtual void update_gpu() = 0;
	#endif
};


class ProcedureCreator{

public:
	ProcedureCreator(const string& classname);
	virtual ~ProcedureCreator(){};

	virtual ProcedureBase* create() = 0;
};

template <class T>
class ProcedureCreatorImpl : public ProcedureCreator{

public:
	ProcedureCreatorImpl<T>(const std::string& classname) : ProcedureCreator(classname) {}
	virtual ~ProcedureCreatorImpl<T>() {}

	virtual ProcedureBase* create() { return new T; }
};

class ProcedureFactory{

public:
	static ProcedureBase* create(const std::string& classname);
	static void registerit(const std::string& classname, ProcedureCreator* creator);
private:
	static std::map<std::string, ProcedureCreator*>& get_table();

};

#define REGISTER(classname) \
	private: \
	static const ProcedureCreatorImpl<classname> creator;\
	static const string proc_name();\
	static const ProcParam get_proc_param(SolverParam solver_param);

#define REGISTERIMPL(classname) \
	const ProcedureCreatorImpl<classname> classname::creator(#classname);\
	const string classname::proc_name(){ return #classname; }\
	const ProcParam classname::get_proc_param(SolverParam solver_param){\
		ProcParam proc_param;\
		bool flag=false;\
		int proc_param_size = solver_param.proc_param_size();\
		for(int i=0; i<proc_param_size; i++){\
			proc_param=solver_param.proc_param(i);\
			if(proc_param.name()==__PROC_NAME__){\
				flag=true;\
				break;\
			}\
		}\
		if(flag == false){\
			LOG(FATAL) << "No parameters specified for `" << __PROC_NAME__ << "' !";\
		}\
		return proc_param;\
	}

#define __PROC_NAME__  \
	proc_name()

}

#endif //__GSBN_PROCEDURE_FACTORY_HPP__
