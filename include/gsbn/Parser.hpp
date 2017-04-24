#include "gsbn/Common.hpp"

#ifndef __GSBN_PARSER_HPP__
#define __GSBN_PARSER_HPP__

namespace gsbn{

/**
 * \class Parser
 * \brief The class analyze the configuration for procedures.
 * 
 * The Parser should be created from a ProcParam type variable which stores all
 * the configuration for a procedure. Parser can analyze the ProcParam structure and
 * return configuration via a string type key.
 */
class Parser{

public:
	/**
	 * \fn Parser()
	 * \brief Constructor
	 * \param proc_param The object which contains the configuration.
	 */
	Parser(ProcParam proc_param);
	/**
	 * \fn ~Parser()
	 * \brief Deconstructor
	 */
	~Parser();
	
	/**
	 * \fn argi()
	 * \brief Look up an integer configuration via key.
	 * \param key The key which indicate the configuration.
	 * \param val The container which hold the value corresponding to the key.
	 * \return True if found, otherwise false.
	 */
	bool argi(const string key, int32_t& val);
	/**
	 * \fn argf()
	 * \brief Look up an float configuration via key.
	 * \param key The key which indicate the configuration.
	 * \param val The container which hold the value corresponding to the key.
	 * \return True if found, otherwise false.
	 */
	bool argf(const string key, float& val);
	/**
	 * \fn args()
	 * \brief Look up an string configuration via key.
	 * \param key The key which indicate the configuration.
	 * \param val The container which hold the value corresponding to the key.
	 * \return True if found, otherwise false.
	 */
	bool args(const string key, string& val);

private:
	ProcParam _proc_param;
};

}

#endif // __GSBN_PARSER_HPP__
