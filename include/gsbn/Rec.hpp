#ifndef __GSBN_REC_HPP__
#define __GSBN_REC_HPP__

#include "Table.hpp"
#include "Database.hpp"

namespace gsbn{

/**
 * \class Rec
 * \bref The recorder of the BCPNN simulation environment.
 *
 * The Rec take snapshot of the database and save it to a binary file.
 */
class Rec{

public:
	
	/**
	 * \fn Rec()
	 * \bref The constructor of Gen.
	 */
	Rec();
	
	/**
	 * \fn init()
	 * \bref Initialize the Rec
	 * \param db The database reference.
	 *
	 * Rec object cannot be used before init() function is called.
	 */
	void init(RecParam rec_param, Database& db);
	void record(bool force=false);

private:
	string _directory;
	bool _enable;
	int _offset;
	int _snapshot_period;
	int _spike_period;
	
	Table *_conf;
	Database* _db;
};

}
#endif //__GSBN_REC_HPP__
