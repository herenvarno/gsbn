#ifndef __GLOBAL_VAR_HPP__
#define __GLOBAL_VAR_HPP__

#include <cstring>
#include <string>
#include <map>

using namespace std;

namespace gsbn{

/**
 * \class GlobalVar
 * \bref The class manages all the global variables which supposed to be seen by
 * every other class.
 * 
 * The GlobalVar maintains a variable pool of std::map type. Variables can be
 * accessed via a string type key. The variables are stored as a unified union type
 * called glvar_t. To easily access global variables of various original types,
 * GlobalVar class defines a set of API dealing with bool, int, float and string
 * type global variables.
 */
class GlobalVar{
public:

	/**
	 * \union glvar_t
	 * \description The global variable entry type. It wraps all common variable type
	 * to a single unified type. Since the size of union type depends on its child
	 * which has the biggest size, the size of glvar_t type is 256 bytes because it
	 * contains a char vector of 256 bytes.
	 */
	typedef union{
		bool b;				/**< boolean type */
		int i;				/**< signed integer type */
		float f;			/**< single floating point type */
		char s[256];	/**< string type */
	}glvar_t;

	/**
	 * \fn get()
	 * \bref Search the glvar_t variable from the pool by a string type key
	 * and return the result.
	 * \param key The key to indicate the global variable.
	 * \param val The glvar_t type container to receive the result.
	 * \return True if found, otherwise false.
	 */
	bool get(const string key, glvar_t& val);
	/**
	 * \fn getb()
	 * \bref Search the bool variable wrapped as glvar_t from the pool by a string type key
	 * and return the result.
	 * \param key The key to indicate the global variable.
	 * \param val The bool type container to receive the result.
	 * \return True if found, otherwise false.
	 */
	bool getb(const string key, bool& val);
	/**
	 * \fn geti()
	 * \bref Search the int variable wrapped as glvar_t from the pool by a string type key
	 * and return the result.
	 * \param key The key to indicate the global variable.
	 * \param val The int type container to receive the result.
	 * \return True if found, otherwise false.
	 */
	bool geti(const string key, int& val);
	/**
	 * \fn getf()
	 * \bref Search the float variable wrapped as glvar_t from the pool by a string type key
	 * and return the result.
	 * \param key The key to indicate the global variable.
	 * \param val The float type container to receive the result.
	 * \return True if found, otherwise false.
	 */
	bool getf(const string key, float& val);
	/**
	 * \fn gets()
	 * \bref Search the string variable wrapped as glvar_t from the pool by a string type key
	 * and return the result.
	 * \param key The key to indicate the global variable.
	 * \param val The string type container to receive the result.
	 * \return True if found, otherwise false.
	 */
	bool gets(const string key, string& val);
	/**
	 * \fn put()
	 * \bref Update the glvar_t variable in the pool located by a string type key
	 * and return the result. If the variable doesn't exist, then create a new one
	 * and associate it with the given key. In either situation, the global variable
	 * will be updated according to the given val.
	 * \param key The key to indicate the global variable.
	 * \param val The value of glvar_t type variable.
	 * \return True if the variable already exists, otherwise false.
	 */
	bool put(const string key, const glvar_t val);
	/**
	 * \fn putb()
	 * \bref Update the bool variable wrapped as glvar_t in the pool located by a string type key
	 * and return the result. If the variable doesn't exist, then create a new one
	 * and associate it with the given key. In either situation, the global variable
	 * will be updated according to the given val.
	 * \param key The key to indicate the global variable.
	 * \param val The value of bool type variable.
	 * \return True if the variable already exists, otherwise false.
	 */
	bool putb(const string key, const bool val);
	/**
	 * \fn puti()
	 * \bref Update the int variable wrapped as glvar_t in the pool located by a string type key
	 * and return the result. If the variable doesn't exist, then create a new one
	 * and associate it with the given key. In either situation, the global variable
	 * will be updated according to the given val.
	 * \param key The key to indicate the global variable.
	 * \param val The value of int type variable.
	 * \return True if the variable already exists, otherwise false.
	 */
	bool puti(const string key, const int val);
	/**
	 * \fn putf()
	 * \bref Update the float variable wrapped as glvar_t in the pool located by a string type key
	 * and return the result. If the variable doesn't exist, then create a new one
	 * and associate it with the given key. In either situation, the global variable
	 * will be updated according to the given val.
	 * \param key The key to indicate the global variable.
	 * \param val The value of float type variable.
	 * \return True if the variable already exists, otherwise false.
	 */
	bool putf(const string key, const float val);
	/**
	 * \fn puts()
	 * \bref Update the string variable wrapped as glvar_t in the pool located by a string type key
	 * and return the result. If the variable doesn't exist, then create a new one
	 * and associate it with the given key. In either situation, the global variable
	 * will be updated according to the given val.
	 * \param key The key to indicate the global variable.
	 * \param val The value of string type variable.
	 * \return True if the variable already exists, otherwise false.
	 */
	bool puts(const string key, const string val);
	
private:
	static map<string, glvar_t> _pool;
};

}

#endif // __GLOBAL_VAR_HPP__
