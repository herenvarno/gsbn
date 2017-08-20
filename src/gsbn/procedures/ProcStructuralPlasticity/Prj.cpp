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
	_wij = db.sync_vector_f32("wij_" + to_string(_id));
	_ej = db.sync_vector_f32("ej_" + to_string(_id));
	
	if(_rank == rank){
		CHECK(_avail_hcu = db.create_sync_vector_i32(".structural_plasticity_avail_hcu_" + to_string(_id)));
		_avail_hcu->resize(_dim_hcu+1);
	}else{
		_avail_hcu = NULL;
	}
	
	shared_buffer_size_list[_rank] += _dim_hcu * _dim_conn;
	
	if(pop_list[_src_pop]._rank==rank){
		CHECK(_local_buffer = db.create_sync_vector_i32(".structural_plasticity_local_buffer_" + to_string(_id)));
		_local_buffer->resize(_dim_hcu*_dim_conn);
		CHECK(_local_buffer1 = db.create_sync_vector_i32(".structural_plasticity_local_buffer1_" + to_string(_id)));
		_local_buffer1->resize(_dim_hcu+1);
	}else{
		_local_buffer=NULL;
		_local_buffer1 = NULL;
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
	
	ptr = _pij->mutable_cpu_data() + row*_dim_mcu;
	# pragma omp parallel for
	for(int i=0; i<_dim_mcu; i++){
		ptr[i] = 0;
	}
	ptr = _eij->mutable_cpu_data() + row*_dim_mcu;
	# pragma omp parallel for
	for(int i=0; i<_dim_mcu; i++){
		ptr[i] = 0;
	}
	ptr = _zj2->mutable_cpu_data() + row*_dim_mcu;
	# pragma omp parallel for
	for(int i=0; i<_dim_mcu; i++){
		ptr[i] = 0;
	}
}

void Prj::get_avail_active_hcu_list(int threshold){
//	vector<int> v;
//	for(int i=0; i<_dim_hcu; i++){
//		v.push_back(i);
//	}
//	return v;

// FIXME : multi-thread bug

	int cnt=0;
	int * ptr_avail_hcu = _avail_hcu->mutable_cpu_data();
	
	const float *ptr_ej = _ej->cpu_data();
	const int *ptr_ii = _ii->cpu_data();
	
	for(int i=0; i<_dim_hcu; i++){
		float ej_acc=0;
		for(int j=0; j<_dim_mcu; j++){
			ej_acc += ptr_ej[i*_dim_mcu+j];
		}
		bool flag=false;
		for(int j=0; j<_dim_conn; j++){
			if(ptr_ii[j]<0){
				flag = true;
				break;
			}
		}
		
		if(ej_acc>threshold && flag==true){
			ptr_avail_hcu[cnt++]=i;
		}
	}
	ptr_avail_hcu[cnt]=-1;
}

vector<int> Prj::prune(int threshold_t, float threshold_wp, float threshold_wn){
	CHECK_GT(threshold_wp, threshold_wn) << "w+ should be larger than w- !!";
	const float* ptr_wij = _wij->cpu_data();
	const int* ptr_ti = _ti->cpu_data();
	const int* ptr_ii = _ii->cpu_data();
	vector<int> v;
	for(int i=0;i<_dim_hcu*_dim_conn; i++){
		if(ptr_ii[i]<0){
			continue;
		}
		if(ptr_ti[i]<threshold_t){
			v.push_back(ptr_ii[i]);
			remove_conn(i);
			continue;
		}
		bool flag=false;
		for(int j=0; j<_dim_mcu; j++){
			if(ptr_wij[i*_dim_mcu+j]>=threshold_wp || ptr_wij[i*_dim_mcu+j]<=threshold_wn){
				flag = true;
				break;
			}
		}
		if(flag==false){
			v.push_back(ptr_ii[i]);
			remove_conn(i);
		}
	}
	if(v.size()<_dim_hcu*_dim_conn){
		v.push_back(-1);
	}
	return v;
}

}
}
