#ifndef __GHCU_POP_HPP__
#define __GHCU_POP_HPP__

#include "Common.hpp"
#include "Table.hpp"
#include "Mcu.hpp"
#include "Hcu.hpp"

namespace ghcu{

/**
 * \class Pop
 * \bref The table class that contains all Populations in the BCPNN network.
 *
 * The shape of Pop table is [HCU_CLASS_POINTER]. The only column in the Pop
 * is the pointer which points to the Hcu that belongs to the specific
 * Population.
 *
 * \warning In current design the Pop table should be used and stored in CPU memory,
 * because the pointer always points to a CPU memory location. Therefore it will
 * be meaningless to synchronize the data to GPU memory and use it.
 */
class Pop : public Table{

public:

	/**
	 * \fn Pop(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);
	 * \bref A simple constructor of class Pop. With initial data.
	 * \param pop_num The number of Pop in the BCPNN.
	 * \param hcu_num The number of Hcu in each Pop.
	 * \param slot_num The total slots for each Hcu.
	 * \param mcu_num The number of Mcu in each Hcu.
	 * \param fanout_num The number of fanout in each Mcu.
	 *
	 * The function not only defines the shape of table, but also fills the table
	 * with \p pop_num rows. If the \p hcu_num is given, The function will call
	 * Pop::append to atomatically allocate a Hcu tables for each Pop and append
	 * pointer of newly allocated Hcu to the Pop table.
	 */
	Pop(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);
	
	/**
	 * \fn ~Pop();
	 * \bref A simple destructor to free allocated Hcu in Pop table.
	 */
	~Pop();
	
	/**
	 * \fn void append(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);
	 * \bref Append Pop to Pop table.
	 * \param pop_num The number of Pop in the BCPNN.
	 * \param hcu_num The number of Hcu in each Pop.
	 * \param slot_num The total slots for each Hcu.
	 * \param mcu_num The number of Mcu in each Hcu.
	 * \param fanout_num The number of fanout in each Mcu.
	 *
	 * The function appends Pop to the existed Pop table
	 * \p pop_num rows. If the \p hcu_num is given, The function will call
	 * Pop::append to atomatically allocate a Hcu tables for each Pop and append
	 * pointer of newly allocated Hcu to the Pop table.
	 */
	void append(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);

};

}



#endif //__GHCU_POP_HPP__
