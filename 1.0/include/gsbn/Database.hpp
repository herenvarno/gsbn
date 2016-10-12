#ifndef __GSBN_DATABASE_HPP__
#define __GSBN_DATABASE_HPP__

#include "gsbn/SyncVector.hpp"
#include "gsbn/Blob.hpp"
#include "gsbn/Table.hpp"

namespace gsbn{

/**
 * \class Database
 * \bref The Database class manage all the tables need by the program. It organize
 * these tables by a hash table (std::map).
 * 
 * The Database class will create these tables and initialize them. It also
 * provides the API to access the tables by their associated names.
 */
class Database{

public:

	/**
	 * \enum proj_idx_t
	 * The index of Table "proj".
	 */
	enum proj_idx_t{
		/** Table index of source population number of a projection */
		IDX_PROJ_SRC_POP,
		/** Table index of destination population number of a projection */
		IDX_PROJ_DEST_POP,
		/** Number of MCU in the destination population. Used to initialize the Pij.
		 * It's redundant information.*/
		IDX_PROJ_MCU_NUM,
		/** Parameter tauzidt.*/
		IDX_PROJ_TAUZIDT,
		/** Parameter tauzjdt.*/
		IDX_PROJ_TAUZJDT,
		/** Parameter tauedt.*/
		IDX_PROJ_TAUEDT,
		/** Parameter taupdt.*/
		IDX_PROJ_TAUPDT,
		/** Parameter eps.*/
		IDX_PROJ_EPS,
		/** Parameter eps2.*/
		IDX_PROJ_EPS2,
		/** Parameter kfti.*/
		IDX_PROJ_KFTI,
		/** Parameter kftj.*/
		IDX_PROJ_KFTJ,
		/** Parameter bgain.*/
		IDX_PROJ_BGAIN,
		/** Parameter wgain.*/
		IDX_PROJ_WGAIN,
		/** Parameter pi0.*/
		IDX_PROJ_PI0,
		IDX_PROJ_COUNT
	};
	/**
	 * \enum pop_idx_t
	 * The index of Table "pop".
	 */
	enum pop_idx_t{
		/** Table index of first HCU in current population */
		IDX_POP_HCU_INDEX,
		/** Table index of HCU number in current population */
		IDX_POP_HCU_NUM,
		IDX_POP_COUNT
	};
	/**
	 * \enum hcu_idx_t
	 * The index of Table "hcu".
	 */
	enum hcu_idx_t{
		/** Table index of first MCU in current HCU */
		IDX_HCU_MCU_INDEX,
		/** Table index of MCU number in current HCU */
		IDX_HCU_MCU_NUM,
		/** Table index of first available incoming projection in current HCU */
		IDX_HCU_ISP_INDEX,
		/** Table index of available incoming projection number in current HCU */
		IDX_HCU_ISP_NUM,
		/** Table index of first available outgoing projection in current HCU */
		IDX_HCU_OSP_INDEX,
		/** Table index of available outgoing projection number in current HCU */
		IDX_HCU_OSP_NUM,
		/** Parameter taumdt */
		IDX_HCU_TAUMDT,
		/** Parameter wtagain */
		IDX_HCU_WTAGAIN,
		/** Parameter maxfqdt */
		IDX_HCU_MAXFQDT,
		/** Parameter igain */
		IDX_HCU_IGAIN,
		/** Parameter wgain */
		IDX_HCU_WGAIN,
		/** Parameter snoise */
		IDX_HCU_SNOISE,
		/** Parameter lgbias */
		IDX_HCU_LGBIAS,
		IDX_HCU_COUNT
	};
	/**
	 * \enum mcu_idx_t
	 * The index of Table "mcu".
	 */
	enum mcu_idx_t{
		/** The index of the first JArray associated with the MCU */
		IDX_MCU_J_ARRAY_INDEX,
		/** The number of JArray rows associated with the MCU */
		IDX_MCU_J_ARRAY_NUM,
		IDX_MCU_COUNT
	};
	/**
	 * \enum hcu_slot_idx_t
	 * The index of Table "hcu_slot".
	 */
	enum hcu_slot_idx_t{
		/** The number of slot owned by the HCU */
		IDX_HCU_SLOT_VALUE,
		IDX_HCU_SLOT_COUNT
	};
	/**
	 * \enum hcu_isp_idx_t
	 * The index of Table "hcu_isp".
	 */
	enum hcu_isp_idx_t{
		/** The projection index */
		IDX_HCU_ISP_VALUE,
		IDX_HCU_ISP_COUNT
	};
	/**
	 * \enum hcu_osp_idx_t
	 * The index of Table "hcu_osp".
	 */
	enum hcu_osp_idx_t{
		/** The projection index */
		IDX_HCU_OSP_VALUE,
		IDX_HCU_OSP_COUNT
	};
	/**
	 * \enum mcu_fanout_idx_t
	 * The index of Table "mcu_fanout".
	 */
	enum mcu_fanout_idx_t{
		/** The number of fanout owned by the MCU */
		IDX_MCU_FANOUT_VALUE,
		IDX_MCU_FANOUT_COUNT
	};
	/**
	 * \enum spk_idx_t
	 * The index of Table "spk".
	 */
	enum spk_idx_t{
		/** The spike generated by the MCU */
		IDX_SPK_VALUE,
		IDX_SPK_COUNT
	};
	/**
	 * \enum epsc_idx_t
	 * The index of Table "epsc".
	 */
	enum epsc_idx_t{
		/** The epsc value, "epsc" table share the same index with "j_array" table */
		IDX_EPSC_VALUE,
		IDX_EPSC_COUNT
	};
	/**
	 * \enum j_array_idx_t
	 * The index of Table "j_array".
	 */
	enum j_array_idx_t{
		/** The Pj */
		IDX_J_ARRAY_PJ,
		/** The Ej */
		IDX_J_ARRAY_EJ,
		/** The Zj */
		IDX_J_ARRAY_ZJ,
		IDX_J_ARRAY_BJ,
		IDX_J_ARRAY_COUNT
	};
	/**
	 * \enum conn_idx_t
	 * The index of Table "conn".
	 */
	enum conn_idx_t{
		/** The source MCU index */
		IDX_CONN_SRC_MCU,
		/** The destination HCU index */
		IDX_CONN_DEST_HCU,
		/** The subprojection index. The subprojection is the index of the incoming
		 * projection (ISP) associated with the destination HCU. Each HCU can have
		 * a certain number of possible ISPs which is predefined by projection table
		 * -- "proj". For example, if there are 4 incoming projections in HCU0, The
		 Database::IDX_CONN_SUBPROJ field then can be an integer from 0 to 3. Though
		 this field, one can look up the real projection index of "proj" table via
		 "hcu" and "isp" table.*/
		IDX_CONN_SUBPROJ,
		/** The projection index. Its the index of entry in "proj" table. It's a
		 * redundant field which can be obtained from Database::IDX_CONN_SUBPROJ. For
		 * convinience, we keep this field.*/
		IDX_CONN_PROJ,
		/** The delay of the connection. \warning Currently it's a constant value. */
		IDX_CONN_DELAY,
		/** The input spike queue associated with the connection. The queue will be
		 * updated in every simulation step via right shifting 1 bit. The MSB of the
		 * value after shifting will indicate the arrival of incoming spike. The
		 * evaluation is before the shifting.*/
		IDX_CONN_QUEUE,
		/** The type of connection. For future use.*/
		IDX_CONN_TYPE,
		/** The first "ij_mat" row associated with the connection. The number of rows
		 * should be determined by the destination HCU's mcu number.*/
		IDX_CONN_IJ_MAT_INDEX,
		IDX_CONN_COUNT
	};
	/**
	 * \enum conn0_idx_t
	 * The index of Table "conn0". similar to Database::conn_idx_t.
	 */
	enum conn0_idx_t{
		IDX_CONN0_SRC_MCU,
		IDX_CONN0_DEST_HCU,
		IDX_CONN0_SUBPROJ,
		IDX_CONN0_PROJ,
		IDX_CONN0_DELAY,
		IDX_CONN0_QUEUE,
		IDX_CONN0_TYPE,
		IDX_CONN0_COUNT
	};
	/**
	 * \enum wij_idx_t
	 * The index of Table "wij".
	 */
	enum wij_idx_t{
		/** The wij value. The "wij" table shares the same index with "ij_mat" table.
		 */
		IDX_WIJ_VALUE,
		IDX_WIJ_COUNT
	};
	/**
	 * \enum i_array_idx_t
	 * The index of Table "i_array".
	 */
	enum i_array_idx_t{
		/** The Pi */
		IDX_I_ARRAY_PI,
		/** The Ei */
		IDX_I_ARRAY_EI,
		/** The Zi */
		IDX_I_ARRAY_ZI,
		/** The Ti */
		IDX_I_ARRAY_TI,
		IDX_I_ARRAY_COUNT
	};
	/**
	 * \enum ij_mat_idx_t
	 * The index of Table "ij_mat".
	 */
	enum ij_mat_idx_t{
		/** The Pij */
		IDX_IJ_MAT_PIJ,
		/** The Eij */
		IDX_IJ_MAT_EIJ,
		/** The Zi2 */
		IDX_IJ_MAT_ZI2,
		/** The Zi2 */
		IDX_IJ_MAT_ZJ2,
		/** The Tij */
		IDX_IJ_MAT_TIJ,
		IDX_IJ_MAT_COUNT
	};
	/**
	 * \enum tmp1_idx_t
	 * The index of Table "tmp1".
	 */
	enum tmp1_idx_t{	// used to record the active spikes
		/** The index of MCU which has generated a spike */
		IDX_TMP1_MCU_IDX,
		IDX_TMP1_COUNT
	};
	/**
	 * \enum tmp2_idx_t
	 * The index of Table "tmp2".
	 */
	enum tmp2_idx_t{	// used to record the update connection whoes I_ARRAY, IJ_MAT and WIJ need to be updated.
		/** The index of connection which receive a incoming spike. */
		IDX_TMP2_CONN,
		/** The destination HCU */
		IDX_TMP2_SRC_MCU,
		/** The destination HCU */
		IDX_TMP2_DEST_HCU,
		/** The subproj */
		IDX_TMP2_SUBPROJ,
		/** The proj */
		IDX_TMP2_PROJ,
		/** The first "ij_mat" row associated with the connection. */
		IDX_TMP2_IJ_MAT_INDEX,
		IDX_TMP2_COUNT
	};
	/**
	 * \enum tmp3_idx_t
	 * The index of Table "tmp3".
	 */
	enum tmp3_idx_t{	// used to record the newly established connection whoes I_ARRAY and IJ_MAT need to be initialized.
		/** The index of connection newly established. */
		IDX_TMP3_CONN,
		/** The destination HCU */
		IDX_TMP3_DEST_HCU,
		/** The fist "ij_mat" row */
		IDX_TMP3_IJ_MAT_IDX,
		/** The initial value should be assign to Pi. */
		IDX_TMP3_PI_INIT,
		/** The initial value should be assign to Pij. */
		IDX_TMP3_PIJ_INIT,
		IDX_TMP3_COUNT
	};
	/**
	 * \enum addr_idx_t
	 * The index of Table "addr".
	 */
	enum addr_idx_t{
		/** The population index associated with the MCU */
		IDX_ADDR_POP,
		/** The HCU index associated with the MCU */
		IDX_ADDR_HCU,
		IDX_ADDR_COUNT
	};
	/**
	 * \enum mode_idx_t
	 * The index of Table "mode".
	 */
	enum mode_idx_t{
		/** The begin time of the mode. */
		IDX_MODE_BEGIN_TIME,
		/** The end time of the mode. */
		IDX_MODE_END_TIME,
		/** The prn to control learning or recall phase. \warning Currently, we only
		 * set prn=0 or 1.*/
		IDX_MODE_PRN,
		IDX_MODE_GAIN_MASK,
		IDX_MODE_PLASTICITY,
		/** The index of stimili. FIXME: need redesign the stimulation procedure.*/
		IDX_MODE_STIM,
		IDX_MODE_COUNT
	};
	/**
	 * \enum stim_idx_t
	 * The index of Table "stim".
	 * \warning The "stim" table store the stimuli regardless the actual width of
	 * the stimuli array which should be larger of equal to the maximun number of
	 * MCUs inside each HCU. The program will NOT perform any check on the stimuli
	 * array provided by user via a simuli file.
	 */
	enum stim_idx_t{
		/** The stimulus value. */
		IDX_STIM_VALUE,
		IDX_STIM_COUNT
	};
	/**
	 * \enum sup_idx_t
	 * The index of Table "sup".
	 */
	enum sup_idx_t{
		/** The dsup. */
		IDX_SUP_DSUP,
		/** The act. */
		IDX_SUP_ACT,
		IDX_SUP_COUNT
	};
	/**
	 * \enum conf_idx_t
	 * The index of Table "conf". FIXME: this table need to be updated.
	 */
	enum conf_idx_t{
		/** The timestamp. */
		IDX_CONF_TIMESTAMP,
		/** The dt, duration time for each step. */
		IDX_CONF_DT,
		/** The prn. */
		IDX_CONF_PRN,
		IDX_CONF_GAIN_MASK,
		IDX_CONF_PLASTICITY,
		/** The stim index. */
		IDX_CONF_STIM,
		IDX_CONF_COUNT
	};
	
	enum rnd_idx_uniform01_t{
		IDX_RND_UNIFORM01_VALUE,
		IDX_RND_UNIFORM01_COUNT
	};
	
	enum rnd_idx_normal_t{
		IDX_RND_NORMAL_VALUE,
		IDX_RND_NORMAL_COUNT
	};

	/**
	 * \fn Database()
	 * \bref A simple constructor of class Database. It creates all the tables.
	 */
	Database();
	/**
	 * \fn ~Database()
	 * \bref A destructor of class Database. It delete all the tables.
	 */
	~Database();
	
	/**
	 * \fn tables()
	 * \bref Get all the tables stored in the database.
	 * \return The vector of tables.
	 */
	vector<Table *> tables();

	
	/**
	 * \fn init_new()
	 * \bref Initialize the tables while creating a new Solver.
	 * \param solver_param The parameters of the solver, provided by user.
	 */
	void init_new(SolverParam solver_param);
	/**
	 * \fn init_copy()
	 * \bref Initialize the tables while copying a Solver from snapshot.
	 * \param solver_state The states of the tables, provided by user.
	 */
	void init_copy(SolverState solver_state);
	
	/**
	 * \fn dump_shapes()
	 * \bref Print the shapes of all tables. For debug.
	 */
	void dump_shapes();
	
	/**
	 * \fn table()
	 * \bref Get a table's pointer by its name.
	 * \param name The name of the table.
	 * \return The pointer to the table.
	 */
	Table* table(const string name);
	void register_table(Table *t);
	
	Blob<int>* blob_i(const string name);
	Blob<float>* blob_f(const string name);
	Blob<double>* blob_d(const string name);
	void register_blob_i(Blob<int> *b);
	void register_blob_f(Blob<float> *b);
	void register_blob_d(Blob<double> *b);
	
	SyncVector<int>* sync_vector_i(const string name);
	SyncVector<float>* sync_vector_f(const string name);
	SyncVector<double>* sync_vector_d(const string name);
	void register_sync_vector_i(const string name, SyncVector<int> *v);
	void register_sync_vector_f(const string name, SyncVector<float> *v);
	void register_sync_vector_d(const string name, SyncVector<double> *v);

private:
	bool _initialized;
	map<string, Table*> _tables;
	map<string, void*> _blobs;
	map<string, void*> _sync_vectors;
};
}

#endif //__GSBN_DATABASE_HPP__
