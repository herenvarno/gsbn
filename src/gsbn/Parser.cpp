#include "gsbn/Parser.hpp"

namespace gsbn{

Parser::Parser(ProcParam proc_param){
	_proc_param = proc_param;
}

Parser::~Parser(){
}

bool Parser::argi(const string key, int32_t& val){
	int size = _proc_param.argi_size();
	for(int i=0; i<size; i++){
		if(_proc_param.argi(i).key() == key){
			val = _proc_param.argi(i).val();
			return true;
		}
	}
	return false;
}

bool Parser::argf(const string key, float& val){
	int size = _proc_param.argf_size();
	for(int i=0; i<size; i++){
		if(_proc_param.argf(i).key() == key){
			val = _proc_param.argf(i).val();
			return true;
		}
	}
	return false;
}

bool Parser::args(const string key, string& val){
	int size = _proc_param.args_size();
	for(int i=0; i<size; i++){
		if(_proc_param.args(i).key() == key){
			val = _proc_param.args(i).val();
			return true;
		}
	}
	return false;
}

}
