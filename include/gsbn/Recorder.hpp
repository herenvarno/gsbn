#ifndef __GSBN_RECORDER_HPP__
#define __GSBN_RECORDER_HPP__

#include "Table.hpp"

namespace gsbn{

class Recorder{

public:
	Recorder(): _directory(), _timestamp(0), _freq(1), _tables(){};
	void set_directory(string directory);
	void set_timestamp(int timestamp);
	void set_freq(int freq);
	void append_table(Table tab);
	void record();

private:
	string _directory;
	int _timestamp;
	int _freq;
	vector<Table> _tables;
};

}
#endif //__GSBN_RECORDER_HPP__
