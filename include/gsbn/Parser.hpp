#include "gsbn/Common.hpp"

#ifndef __GSBN_PARSER_HPP__
#define __GSBN_PARSER_HPP__

namespace gsbn{

class Parser{

public:
	Parser(ProcParam proc_param);
	~Parser();
	
	bool argi(const string key, int32_t& val);
	bool argf(const string key, float& val);
	bool args(const string key, string& val);

private:
	ProcParam _proc_param;
};

}

#endif // __GSBN_PARSER_HPP__
