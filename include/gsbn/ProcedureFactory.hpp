#ifndef __GSBN_PROCEDURE_FACTORY_HPP__
#define __GSBN_PROCEDURE_FACTORY_HPP__

#include "gsbn/Database.hpp"

namespace gsbn{

class ProcedureBase{

public:
	ProcedureBase() : _initialized(false){};
	virtual ~ProcedureBase(){};
	
	virtual void init_new(NetParam net_param, Database& db) = 0;
	virtual void init_copy(Database& db) = 0;
	virtual void update_cpu() = 0;
	#ifndef CPU_ONLY
	virtual void update_gpu() = 0;
	#endif
	
	bool initilized(){return _initialized;}
private:
	bool _initialized;
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
	static const ProcedureCreatorImpl<classname> creator;

#define REGISTERIMPL(classname) \
	const ProcedureCreatorImpl<classname> classname::creator(#classname);

}

#endif //__GSBN_PROCEDURE_FACTORY_HPP__
