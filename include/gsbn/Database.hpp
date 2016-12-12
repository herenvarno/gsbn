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
		IDX_CONF_OLD_PRN,
		IDX_CONF_GAIN_MASK,
		IDX_CONF_PLASTICITY,
		/** The stim index. */
		IDX_CONF_STIM,
		IDX_CONF_MODE,
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
	void init_copy(SolverParam solver_param, SolverState solver_state);
	
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
	Table* create_table(const string name, const vector<int> fields);
	void register_table(const string name, Table *t);
	
	Blob<int>* blob_i(const string name);
	Blob<float>* blob_f(const string name);
	Blob<double>* blob_d(const string name);
	void register_blob_i(Blob<int> *b);
	void register_blob_f(Blob<float> *b);
	void register_blob_d(Blob<double> *b);
	
	SyncVector<int8_t>* create_sync_vector_i8(const string name);
	SyncVector<int16_t>* create_sync_vector_i16(const string name);
	SyncVector<int32_t>* create_sync_vector_i32(const string name);
	SyncVector<int64_t>* create_sync_vector_i64(const string name);
	SyncVector<fp16>* create_sync_vector_f16(const string name);
	SyncVector<float>* create_sync_vector_f32(const string name);
	SyncVector<double>* create_sync_vector_f64(const string name);
	
	SyncVector<int8_t>* sync_vector_i8(const string name);
	SyncVector<int16_t>* sync_vector_i16(const string name);
	SyncVector<int32_t>* sync_vector_i32(const string name);
	SyncVector<int64_t>* sync_vector_i64(const string name);
	SyncVector<fp16>* sync_vector_f16(const string name);
	SyncVector<float>* sync_vector_f32(const string name);
	SyncVector<double>* sync_vector_f64(const string name);
	
	void register_sync_vector_i8(const string name, SyncVector<int8_t> *v);
	void register_sync_vector_i16(const string name, SyncVector<int16_t> *v);
	void register_sync_vector_i32(const string name, SyncVector<int32_t> *v);
	void register_sync_vector_i64(const string name, SyncVector<int64_t> *v);
	void register_sync_vector_f16(const string name, SyncVector<fp16> *v);
	void register_sync_vector_f32(const string name, SyncVector<float> *v);
	void register_sync_vector_f64(const string name, SyncVector<double> *v);
	
	SolverState state_to_proto();

private:
	bool _initialized;
	map<string, Table*> _tables;
	map<string, void*> _blobs;
	map<string, void*> _sync_vectors;
};
}

#endif //__GSBN_DATABASE_HPP__

