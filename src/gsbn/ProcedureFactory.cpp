#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{

ProcedureCreator::ProcedureCreator(const string& classname){
	ProcedureFactory::registerit(classname, this);
}

void ProcedureFactory::registerit(const std::string& classname, ProcedureCreator* creator)
{
	get_table()[classname] = creator;
}

ProcedureBase* ProcedureFactory::create(const std::string& classname)
{
	std::map<std::string, ProcedureCreator*>::iterator i;
	i = get_table().find(classname);

	if(i != get_table().end())
		return i->second->create();
	else
		return (ProcedureBase*)NULL;
}

std::map<std::string, ProcedureCreator*>& ProcedureFactory::get_table()
{
	static std::map<std::string, ProcedureCreator*> table;
	return table;
}

}
