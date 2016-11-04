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
	void init(Database& db);
	/**
	 * \fn set_directory()
	 * \param directory The directory.
	 * \bref Set the directory to save snapshots.
	 */
	void set_directory(string directory);
	/**
	 * \fn set_period()
	 * \param period The period.
	 * \bref Set the period of the generation of snapshot, the period equals
	 * to the number of simulation steps between 2 snapshots.
	 */
	void set_period(int period);
	/**
	 * \fn record()
	 * \param force The flag which decides whether to ignore the period while
	 * generating the snapshot. If it's True, The snapshot will be generated even
	 * the timestamp doesn't meet the correct generating time.
	 * \bref Generate and save the snapshot file.
	 */
	void record(bool force=false);

private:
	string _directory;
	int _period;
	Table *_conf;
	Database* _db;
};

}
#endif //__GSBN_REC_HPP__
