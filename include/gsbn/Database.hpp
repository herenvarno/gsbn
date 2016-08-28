#ifndef __GSBN_DATABASE_HPP__
#define __GSBN_DATABASE_HPP__

#include "gsbn/Table.hpp"

namespace gsbn{

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
		/** Table index of first available outgoing projection in current HCU */
		IDX_HCU_SUBPROJ_INDEX,
		/** Table index of available outgoing projection number in current HCU */
		IDX_HCU_SUBPROJ_NUM,
		IDX_HCU_COUNT
	};
	enum mcu_idx_t{
		IDX_MCU_J_ARRAY_INDEX,
		IDX_MCU_J_ARRAY_NUM,
		IDX_MCU_COUNT
	};
	enum hcu_slot_idx_t{
		IDX_HCU_SLOT_VALUE,
		IDX_HCU_SLOT_COUNT
	};
	enum hcu_subproj_idx_t{
		IDX_HCU_SUBPROJ_VALUE,
		IDX_HCU_SUBPROJ_COUNT
	};
	enum mcu_fanout_idx_t{
		IDX_MCU_FANOUT_VALUE,
		IDX_MCU_FANOUT_COUNT
	};
	enum spk_idx_t{
		IDX_SPK_VALUE,
		IDX_SPK_COUNT
	};
	enum epsc_idx_t{
		IDX_EPSC_VALUE,
		IDX_EPSC_COUNT
	};
	enum j_array_idx_t{
		IDX_J_ARRAY_PJ,
		IDX_J_ARRAY_EJ,
		IDX_J_ARRAY_ZJ,
		IDX_J_ARRAY_COUNT
	};
	enum conn_idx_t{
		IDX_CONN_SRC_MCU,
		IDX_CONN_DEST_HCU,
		IDX_CONN_DEST_SUBPROJ,
		IDX_CONN_DELAY,
		IDX_CONN_QUEUE,
		IDX_CONN_TYPE,
		IDX_CONN_IJ_MAT_INDEX,
		IDX_CONN_COUNT
	};
	enum conn0_idx_t{
		IDX_CONN0_SRC_MCU,
		IDX_CONN0_DEST_HCU,
		IDX_CONN0_DEST_SUBPROJ,
		IDX_CONN0_DELAY,
		IDX_CONN0_QUEUE,
		IDX_CONN0_TYPE,
		IDX_CONN0_COUNT
	};
	enum wij_idx_t{
		IDX_WIJ_VALUE,
		IDX_WIJ_COUNT
	};
	enum i_array_idx_t{
		IDX_I_ARRAY_PI,
		IDX_I_ARRAY_EI,
		IDX_I_ARRAY_ZI,
		IDX_I_ARRAY_TI,
		IDX_I_ARRAY_COUNT
	};
	enum ij_mat_idx_t{
		IDX_IJ_MAT_PIJ,
		IDX_IJ_MAT_EIJ,
		IDX_IJ_MAT_ZI2,
		IDX_IJ_MAT_ZJ2,
		IDX_IJ_MAT_TIJ,
		IDX_IJ_MAT_COUNT
	};
	enum tmp1_idx_t{
		IDX_TMP1_MCU_IDX,
		IDX_TMP1_COUNT
	};
	enum tmp2_idx_t{
		IDX_TMP2_CONN,
		IDX_TMP2_DEST_MCU,
		IDX_TMP2_DEST_SUBPROJ,
		IDX_TMP2_COUNT
	};
	enum tmp3_idx_t{
		IDX_TMP3_CONN,
		IDX_TMP3_DEST_HCU,
		IDX_TMP3_COUNT
	};
	enum addr_idx_t{
		IDX_ADDR_POP,
		IDX_ADDR_HCU,
		IDX_ADDR_COUNT
	};
	enum mode_idx_t{
		IDX_MODE_BEGIN_TIME,
		IDX_MODE_END_TIME,
		IDX_MODE_TYPE,
		IDX_MODE_STIM,
		IDX_MODE_COUNT
	};
	enum stim_idx_t{
		IDX_STIM_VALUE,
		IDX_STIM_COUNT
	};
	enum sup_idx_t{
		IDX_SUP_DSUP,
		IDX_SUP_ACT,
		IDX_SUP_COUNT
	};
	enum conf_idx_t{
		IDX_CONF_KP,
		IDX_CONF_KE,
		IDX_CONF_KZJ,
		IDX_CONF_KZI,
		IDX_CONF_KFTJ,
		IDX_CONF_KFTI,
		IDX_CONF_BGAIN,
		IDX_CONF_WGAIN,
		IDX_CONF_WTAGAIN,
		IDX_CONF_IGAIN,
		IDX_CONF_EPS,
		IDX_CONF_LGBIAS,
		IDX_CONF_SNOISE,
		IDX_CONF_MAXFQDT,
		IDX_CONF_TAUMDT,
		IDX_CONF_COUNT
	};

	Database();
	~Database();
	
	vector<Table *> tables();
	Table* table(string name);
	
	void init_new(SolverParam solver_param);
	void init_copy(SolverState solver_state);
	
	void dump_shapes();

private:
	bool _initialized;
	map<string, Table*> _tables;
};
}

#endif //__GSBN_DATABASE_HPP__
