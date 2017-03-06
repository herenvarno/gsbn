#ifndef __GSBN_DATABASE_HPP__
#define __GSBN_DATABASE_HPP__

#include "gsbn/SyncVector.hpp"

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
	map<string, void*> _sync_vectors;
};
}

#endif //__GSBN_DATABASE_HPP__

