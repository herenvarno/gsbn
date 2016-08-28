#ifndef __GSBN_REC_HPP__
#define __GSBN_REC_HPP__

#include "Table.hpp"

namespace gsbn{

class Rec{

public:
	Rec(): _directory(), _period(1), _tables(){};
	void set_directory(string directory);
	void set_period(int period);
	void append_tables(vector<Table *> tabs);
	void record(int timestamp, bool force=false);

private:
	string _directory;
	int _period;
	vector<Table*> _tables;
};

}
#endif //__GSBN_REC_HPP__
