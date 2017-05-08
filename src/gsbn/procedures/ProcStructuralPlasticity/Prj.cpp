#include "gsbn/procedures/ProcStructuralPlasticity/Prj.hpp"

namespace gsbn{
namespace proc_structural_plasticity{

Prj::Prj(int& id, vector<int>& shared_buffer_size_list, vector<Pop>& pop_list, ProjParam prj_param, Database& db, int rank){
	_id = id;
	_src_pop = prj_param.src_pop();
	_dest_pop = prj_param.dest_pop();
	_dim_hcu = pop_list[_dest_pop]._dim_hcu;
	_dim_mcu = pop_list[_dest_pop]._dim_mcu;
	_dim_conn = pop_list[_src_pop]._dim_hcu * pop_list[_src_pop]._dim_mcu;
	if(_dim_conn > prj_param.slot_num()){
		_dim_conn = prj_param.slot_num();
	}
	_rank = pop_list[_dest_pop]._rank;
	_shared_buffer_offset = shared_buffer_size_list[_rank];
	
	// DO NOT CHECK THE RETURN VALUE, SINCE THE VECTORS MAYBE NOT IN THE CURRENT
	// RANK.
	_ii = db.sync_vector_i32("ii_" + to_string(_id));
	_di = db.sync_vector_i32("di_" + to_string(_id));
	_ti = db.sync_vector_i32("ti_" + to_string(_id));
	_pi = db.sync_vector_f32("pi_" + to_string(_id));
	_ei = db.sync_vector_f32("ei_" + to_string(_id));
	_zi = db.sync_vector_f32("zi_" + to_string(_id));
	_pij = db.sync_vector_f32("pij_" + to_string(_id));
	_eij = db.sync_vector_f32("eij_" + to_string(_id));
	_zj2 = db.sync_vector_f32("zj2_" + to_string(_id));
	
	shared_buffer_size_list[_rank] += _dim_hcu * _dim_conn;
	
	if(pop_list[_src_pop]._rank==rank){
		CHECK(_local_buffer = db.create_sync_vector_i32(".structural_plasticity_local_buffer_" + to_string(_id)));
		_local_buffer->resize(_dim_hcu*_dim_conn);
	}else{
		_local_buffer=NULL;
	}
	
	id++;
}

Prj::~Prj(){
}

void Prj::assign_conn(int *ptr_new_ii, int *ptr_new_di){
	int *ptr_ii = _ii->mutable_cpu_data();
	int *ptr_di = _di->mutable_cpu_data();
	
	for(int i=0; i<_dim_hcu * _dim_conn; i++){
		ptr_ii[i] = ptr_new_ii[i];
		ptr_di[i] = ptr_new_di[i];
	}
}

void Prj::remove_conn(int row){
	(_ii->mutable_cpu_data())[row] = -1;
	(_di->mutable_cpu_data())[row] = 0;
	(_ti->mutable_cpu_data())[row] = 0;
	(_pi->mutable_cpu_data())[row] = 0;
	(_ei->mutable_cpu_data())[row] = 0;
	
	float *ptr;
	
	ptr = _pij->mutable_cpu_data() + row;
	# pragma omp parallel for
	for(int i=0; i<_dim_mcu; i++){
		ptr[i] = 0;
	}
	ptr = _eij->mutable_cpu_data() + row;
	# pragma omp parallel for
	for(int i=0; i<_dim_mcu; i++){
		ptr[i] = 0;
	}
	ptr = _zj2->mutable_cpu_data() + row;
	# pragma omp parallel for
	for(int i=0; i<_dim_mcu; i++){
		ptr[i] = 0;
	}
}


}
}
