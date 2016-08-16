#ifndef __GSBN_SPIKE_HPP__
#define __GSBN_SPIKE_HPP__

#include "Table.hpp"

namespace gsbn{

/**
 * \class Spike
 * \bref The table class that contains all spikes generated by MCUs.
 *
 * The shape of Spike table is [SPIKE_GEN]. There is only one column
 * which indicates the whether the MCU has generated a spike in current cycle.
 * 
 * \Warning The class need to be synchronized in each cycle. So, we keep it
 * simple to reduce the amount of data to be exchanged.
 */
class Spike: public Table{

public:

	/**
	 * \fn Spike()
	 * \bref A simple constructor of class Spike. With initial data.
	 * \param mcu_num The number of rows to be initialized.
	 * \param fanout_num The initial spike, default is 0 (No spike).
	 *
	 * The function not only defines the shape of table, but also fills the table
	 * with mcu_num rows, and initializes the SPIKE_GEN column with
	 * init_spike.
	 */
	Spike(int mcu_num=0, unsigned char init_spike=0);
	
	/**
	 * \fn append()
	 * \bref Append new rows to the table and initialize them.
	 * \param mcu_num The number of rows to be initialized.
	 * \param fanout_num The initial spike, default is 0 (No spike).
	 *
	 * The function append new rows to the existed Spike table and initialize
	 * these rows.
	 */
	void append(int mcu_num, unsigned char init_spike=0);
	
	/**
	 * \fn dump()
	 * \bref Dump the memory data to a string.
	 * \return The string.
	 */
	const string dump();

};

}

#endif //__GSBN_MCU_FANOUT_HPP__
