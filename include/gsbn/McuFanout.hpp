#ifndef __GSBN_MCU_FANOUT_HPP__
#define __GSBN_MCU_FANOUT_HPP__

#include "Table.hpp"

namespace gsbn{

/**
 * \class McuFanout
 * \bref The table class that contains all available fanout for all MCUs.
 *
 * The shape of McuFanout table is [AVAILABLE_FANOUT]. There is only one column
 * which indicates the number of connections this MCU can still establish.
 */
class McuFanout: public Table{

public:

	/**
	 * \fn McuFanout()
	 * \bref A simple constructor of class MCU. With initial data.
	 * \param mcu_num The number of rows to be initialized.
	 * \param fanout_num The total fanout for each MCU.
	 *
	 * The function not only defines the shape of table, but also fills the table
	 * with mcu_num rows, and initializes the AVAILABLE_FANOUT column with
	 * fanout_num.
	 */
	McuFanout(int mcu_num=0, int fanout_num=0);
	
	/**
	 * \fn append()
	 * \bref Append new rows to the table and initialize them.
	 * \param mcu_num The number of rows to be initialized.
	 * \param fanout_num The total fanout for each MCU.
	 *
	 * The function append new rows to the existed McuFanout table and initialize
	 * these rows.
	 */
	void append(int mcu_num, int fanout_num=0);
	
	/**
	 * \fn add_fanout()
	 * \bref A function which modify the available fanout of a specific row/MCU.
	 * \param index The index corresponding to the specific row.
	 * \param fanout_num The number of fanout to be added. It can be any integer,
	 * including positive integer, negative integer as well as zero.
	 * 
	 * The function modifies the fanout number of a specific row. Note that, it will
	 * not always "increase" the fanout number because the fanout_num can be
	 * negative which will make the function acting like McuFanout::del_fanout.
	 */
	void add_fanout(int index, int fanout_num);
	
	/**
	 * \fn del_fanout()
	 * \bref A function which modify the available fanout of a specific row/MCU.
	 * \param index The index corresponding to the specific row.
	 * \param fanout_num The number of fanout to be added. It can be any integer,
	 * including positive integer, negative integer as well as zero.
	 *
	 * The function is just a wrap of McuFanout::add_fanout. The only difference is that
	 * when fanout_num is positive, it will "decrease" the fanout number.
	 */
	void del_fanout(int index, int fanout_num);

};

}

#endif //__GSBN_MCU_FANOUT_HPP__
