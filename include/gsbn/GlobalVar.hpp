#include <map>

namespace gsbn{

typedef union{
	bool b,
	int i,
	float f,
	char s[256]
}glvar_t;

class GlobalVarPool{
public:
	bool get(const string key, glvar_t& val);
	bool getb(const string key, bool& val);
	bool geti(const string key, int& val);
	bool getf(const string key, float& val);
	bool gets(const string key, string& val);
	
	bool put(const string key, const glvar_t val);
	bool putb(const string key, const bool val);
	bool puti(const string key, const int val);
	bool putf(const string key, const float val);
	bool puts(const string key, const string val);
	
private:
	static map<string, glvar_t> _pool;
};

}
