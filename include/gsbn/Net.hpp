#ifndef __GSBN_NET_HPP__
#define __GSBN_NET_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/Random.hpp"

namespace gsbn{

#define __DELAY__ 1

/**
 * \class Net
 * \bref The class Net describe the BCPNN network.
 *
 * The Net works in 2 phases: init() and update(). The init() phase runs once after
 * the class is been built, while the update() phase runs in every simulation step.
 */
class Net{

public:
	/**
	 * \fn Net()
	 * \bref The constructor of the class Net. The Net object shouldn't be used until
	 * the init() function is called.
	 */
	Net();
	
	/**
	 * \fn init()
	 * \bref Initialize the Net based on the Database.
	 */
	void init(Database& db);
	
	/**
	 * \fn update()
	 * \bref Update the state in database for each simulation step.
	 *
	 * The update procedure consists of many phases. Currently it has 12 phases.
	 *
	 * \warning In future, this function will be redesigned for dynamic registration
	 * of update phases.
	 */
	void update();

protected:
	void update_phase_0();
	/**
	 * \fn update_phase_1()
	 * \bref Update the j_array. (Pj, Ej, Zj, EPSC, DSUP). The kernel function will
	 * execute NUM_MCU * NUM_PROJECTION_IN_EACH_HCU times.
	 */
	void update_phase_1();
	void update_phase_2();
	/**
	 * \fn update_phase_3()
	 * \bref Halfnormalization. The kernel function will
	 * execute NUM_HCU times.
	 */
	void update_phase_3();
	/**
	 * \fn update_phase_4()
	 * \bref Generating spike. The kernel function will
	 * execute NUM_MCU times.
	 *
	 * \warning This function need to be updated due to the mechanism of stimulation.
	 */
	void update_phase_4();
	/**
	 * \fn update_phase_5()
	 * \bref Generate short spike list in table "tmp1". The kernel function will
	 * execute NUM_HCU times. It can't be merged to update_phase4() because this
	 * phase suppose to execute on CPU, since it needs conflict control.
	 */
	void update_phase_5();
	/**
	 * \fn update_phase_6()
	 * \bref Increase Zj based on "tmp1".
	 * The kernel function will
	 * execute NUM_TMP1 times.
	 */
	void update_phase_6();
	/**
	 * \fn update_phase_7()
	 * \bref Generate short list for active connections after detecting a incoming
	 * spike in "tmp2".
	 * The kernel function will execute NUM_CONN times.
	 */
	void update_phase_7();
	/**
	 * \fn update_phase_8()
	 * \bref Based on "tmp2", update IArray, IJMat, Wij, update EPSC
	 * The kernel function will execute NUM_TMP2 times for updating IArray, IJMat
	 and Wij. For updating EPSC, it will loop NUM_TMP2 times on CPU.
	 */
	void update_phase_8();
	/**
	 * \fn update_phase_9()
	 * \bref Special spike, "REQ" and "ACK". If a new connection is required,
	 * establish the connetion by appending a new row in "conn" table and "tmp3"
	 * table. "tmp3" table is used for initializing the new rows of IArray, IJMat.
	 * It will loop NUM_CONN0 times, execute on CPU.
	 */
	void update_phase_9();
	/**
	 * \fn update_phase_10()
	 * \bref Initialize new rows in IArray and IJMat based on "tmp3".
	 * The kernel function will execute NUM_TMP3 times.
	 */
	void update_phase_10();
	/**
	 * \fn update_phase_11()
	 * \bref Send special spike based on "tmp1" by inserting new rows to the
	 * "conn0" table.
	 * It will loop NUM_TMP1 times, execute on CPU.
	 */
	void update_phase_11();
	/**
	 * \fn update_phase_12()
	 * \bref Increase Zj2 based on "tmp1".
	 * It first generate the "tmp2" table for those MCUs which has a outgoing
	 * spike and connections associated with the MCUs. Then the kernel function will
	 * execute NUM_TMP2 times for increaing Zj2.
	 *
	 * \warning Zi and Zi2 are already increased in update_phase_8().
	 * \warning Here we reused the "tmp2", it should be updated to use a different
	 * temporary table. In this way we can get clear information for debugging.
	 */
	void update_phase_12();
	void update_phase_13();
	
	#ifndef CPU_ONLY
	void update_phase_0_gpu();
	void update_phase_1_gpu();
//	void update_phase_2_gpu();
	void update_phase_3_gpu();
	void update_phase_4_gpu();
	void update_phase_5_gpu();
	void update_phase_6_gpu();
	void update_phase_7_gpu();
	void update_phase_8_gpu();
	void update_phase_9_gpu();
	void update_phase_10_gpu();
	void update_phase_11_gpu();
	void update_phase_12_gpu();
	#endif

private:
	Random _rnd;	
	
	Table* _j_array;
	Table* _spk;
	Table* _hcu;
	Table* _sup;
	Table* _stim;
	Table* _mcu;
	Table* _tmp1;
	Table* _epsc;
	Table* _conf;
	Table* _addr;
	Table* _proj;
	Table *_pop;
	Table *_mcu_fanout;
	Table *_hcu_slot;
	Table *_i_array;
	Table *_ij_mat;
	Table *_wij;
	Table *_tmp2;
	Table *_tmp3;
	Table *_conn;
	Table *_conn0;
	Table *_hcu_isp;
	Table *_hcu_osp;
	Table *_rnd_uniform01;
	Table *_rnd_normal;
	
	vector<int> _empty_conn0_list;
	vector<vector<int>> _existed_conn_list;

};

}
#endif //_GSBN_NET_HPP__
