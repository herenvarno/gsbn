#ifndef __GSBN_DATABASE_HPP__
#define __GSBN_DATABASE_HPP__

#include "gsbn/SyncVector.hpp"

namespace gsbn{

/**
 * \class Database
 * \brief The Database class manage all the SyncVector need by the program. It organize
 * these vectors by a ordered map (std::map).
 * 
 */
class Database{

public:

	/**
	 * \fn Database()
	 * \brief A simple constructor of class Database.
	 */
	Database();
	/**
	 * \fn ~Database()
	 * \brief A destructor of class Database. It will destroy all the stored data.
	 * and realise the memory usage.
	 */
	~Database();

	
	/**
	 * \fn init_new()
	 * \brief Initialize an empty Database object.
	 * \param solver_param The parameters of the solver.
	 */
	void init_new(SolverParam solver_param);
	/**
	 * \fn init_copy()
	 * \brief Initialize a Database object with privided Solver state.
	 * \param solver_param The parameters of the solver.
	 * \param solver_state The states of the solver, it contains the actual data of
	 * a Database snapshot.
	 */
	void init_copy(SolverParam solver_param, SolverState solver_state);
	
	/**
	 * \fn create_sync_vector_i8()
	 * \brief Create a SyncVector of type int8.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the newly created SyncVector or NULL if the vector is
	 * already existed.
	 */
	SyncVector<int8_t>* create_sync_vector_i8(const string name);
	/**
	 * \fn create_sync_vector_i16()
	 * \brief Create a SyncVector of type int16.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the newly created SyncVector or NULL if the vector is
	 * already existed.
	 */
	SyncVector<int16_t>* create_sync_vector_i16(const string name);
	/**
	 * \fn create_sync_vector_i32()
	 * \brief Create a SyncVector of type int32.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the newly created SyncVector or NULL if the vector is
	 * already existed.
	 */
	SyncVector<int32_t>* create_sync_vector_i32(const string name);
	/**
	 * \fn create_sync_vector_i64()
	 * \brief Create a SyncVector of type int64.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the newly created SyncVector or NULL if the vector is
	 * already existed.
	 */
	SyncVector<int64_t>* create_sync_vector_i64(const string name);
	/**
	 * \fn create_sync_vector_f16()
	 * \brief Create a SyncVector of type half float.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the newly created SyncVector or NULL if the vector is
	 * already existed.
	 */
	SyncVector<fp16>* create_sync_vector_f16(const string name);
	/**
	 * \fn create_sync_vector_f32()
	 * \brief Create a SyncVector of type float.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the newly created SyncVector or NULL if the vector is
	 * already existed.
	 */
	SyncVector<float>* create_sync_vector_f32(const string name);
	/**
	 * \fn create_sync_vector_f64()
	 * \brief Create a SyncVector of type double float.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the newly created SyncVector or NULL if the vector is
	 * already existed.
	 */
	SyncVector<double>* create_sync_vector_f64(const string name);
	
	/**
	 * \fn sync_vector_i8()
	 * \brief Get a SyncVector of type int8 from the Database via its name.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the SyncVector or NULL if the vector is
	 * not existed.
	 */
	SyncVector<int8_t>* sync_vector_i8(const string name);
	/**
	 * \fn sync_vector_i16()
	 * \brief Get a SyncVector of type int16 from the Database via its name.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the SyncVector or NULL if the vector is
	 * not existed.
	 */
	SyncVector<int16_t>* sync_vector_i16(const string name);
		/**
	 * \fn sync_vector_i32()
	 * \brief Get a SyncVector of type int32 from the Database via its name.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the SyncVector or NULL if the vector is
	 * not existed.
	 */
	SyncVector<int32_t>* sync_vector_i32(const string name);
	/**
	 * \fn sync_vector_i64()
	 * \brief Get a SyncVector of type int64 from the Database via its name.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the SyncVector or NULL if the vector is
	 * not existed.
	 */
	SyncVector<int64_t>* sync_vector_i64(const string name);
	/**
	 * \fn sync_vector_f16()
	 * \brief Get a SyncVector of type half float from the Database via its name.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the SyncVector or NULL if the vector is
	 * not existed.
	 */
	SyncVector<fp16>* sync_vector_f16(const string name);
	/**
	 * \fn sync_vector_f32()
	 * \brief Get a SyncVector of type float from the Database via its name.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the SyncVector or NULL if the vector is
	 * not existed.
	 */
	SyncVector<float>* sync_vector_f32(const string name);
	/**
	 * \fn sync_vector_f64()
	 * \brief Get a SyncVector of type double float from the Database via its name.
	 * \param name The unique name associated with the SyncVector.
	 * \return The pointer of the SyncVector or NULL if the vector is
	 * not existed.
	 */
	SyncVector<double>* sync_vector_f64(const string name);
	
	/**
	 * \fn register_sync_vector_i8()
	 * \brief Register a SyncVector to the Database.
	 * \param name The unique name associated with the SyncVector.
	 * \param v The SyncVector object.
	 */
	void register_sync_vector_i8(const string name, SyncVector<int8_t> *v);
	/**
	 * \fn register_sync_vector_i16()
	 * \brief Register a SyncVector to the Database.
	 * \param name The unique name associated with the SyncVector.
	 * \param v The SyncVector object.
	 */
	void register_sync_vector_i16(const string name, SyncVector<int16_t> *v);
	/**
	 * \fn register_sync_vector_i32()
	 * \brief Register a SyncVector to the Database.
	 * \param name The unique name associated with the SyncVector.
	 * \param v The SyncVector object.
	 */
	void register_sync_vector_i32(const string name, SyncVector<int32_t> *v);
	/**
	 * \fn register_sync_vector_i64()
	 * \brief Register a SyncVector to the Database.
	 * \param name The unique name associated with the SyncVector.
	 * \param v The SyncVector object.
	 */
	void register_sync_vector_i64(const string name, SyncVector<int64_t> *v);
	/**
	 * \fn register_sync_vector_f16()
	 * \brief Register a SyncVector to the Database.
	 * \param name The unique name associated with the SyncVector.
	 * \param v The SyncVector object.
	 */
	void register_sync_vector_f16(const string name, SyncVector<fp16> *v);
	/**
	 * \fn register_sync_vector_f32()
	 * \brief Register a SyncVector to the Database.
	 * \param name The unique name associated with the SyncVector.
	 * \param v The SyncVector object.
	 */
	void register_sync_vector_f32(const string name, SyncVector<float> *v);
	/**
	 * \fn register_sync_vector_f64()
	 * \brief Register a SyncVector to the Database.
	 * \param name The unique name associated with the SyncVector.
	 * \param v The SyncVector object.
	 */
	void register_sync_vector_f64(const string name, SyncVector<double> *v);
	
	/**
	 * \fn state_to_proto()
	 * \brief Dump the Database snapshot to Protobuf format object SolverState.
	 */
	SolverState state_to_proto();

private:
	bool _initialized;
	map<string, void*> _sync_vectors;
};
}

#endif //__GSBN_DATABASE_HPP__

