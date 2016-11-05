# include "gsbn/procedures/ProcNetGroup/Group.hpp"

namespace gsbn{
namespace proc_net_group{

void Group::init_new(HcuParam hcu_param, Database& db, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){
	CHECK(list_group);
	CHECK(list_hcu);
	CHECK(list_conn);

	_id = list_group->size();
	list_group->push_back(this);
	
	CHECK(_spike = db.sync_vector_i("spike"));
	CHECK(_rnd_uniform01 = db.sync_vector_f(".rnd_uniform01"));
	CHECK(_rnd_normal = db.sync_vector_f(".rnd_normal"));
	CHECK(_lginp = db.sync_vector_f(".lginp"));
	CHECK(_wmask = db.sync_vector_f(".wmask"));
	CHECK(_conf = db.table(".conf"));	

	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];

	_hcu_start = list_hcu->size();
	_mcu_start = _spike->cpu_vector()->size();
	_hcu_num = hcu_param.hcu_num();
	_mcu_num=0;
	for(int i=0; i<_hcu_num; i++){
		Hcu* h = new Hcu();
		h->init_new(hcu_param, db, list_hcu, list_conn, msg);
		h->_group_id=_id;
		_mcu_num += h->_mcu_num;
	}
	
	CHECK(_dsup = db.create_sync_vector_f("dsup_"+to_string(_id)));
	_dsup->mutable_cpu_vector()->resize(_mcu_num, 0);
	CHECK(_act = db.create_sync_vector_f("act_"+to_string(_id)));
	_act->mutable_cpu_vector()->resize(_mcu_num, 0);
	
	_taumdt = dt/hcu_param.taum();
	_wtagain = hcu_param.wtagain();
	_maxfqdt = hcu_param.maxfq()*dt;
	_igain = hcu_param.igain();
	_wgain = hcu_param.wgain();
	_lgbias = hcu_param.lgbias();
	_conn_num = 0;
	
}

void Group::init_copy(HcuParam hcu_param, Database& db, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){
}


void update_dsup_kernel_cpu(
	int i,
	int j,
	int dim_x,
	int dim_y,
	int dim_z,
	int mcu_num_in_pop,
	const float* ptr_epsc,
	const float* ptr_bj,
	const float* ptr_lginp,
	const float* ptr_wmask,
	const float* ptr_rnd_normal,
	float* ptr_dsup,
	float wgain,
	float lgbias,	
	float igain,
	float taumdt
){
	int idx = i*dim_z+j;
	float wsup=0;
	int offset=0;
	for(int m=0; m<dim_x; m++){
		wsup += ptr_bj[offset+idx] + ptr_epsc[offset+idx];
		offset += mcu_num_in_pop;
	}
	float sup = lgbias + igain * ptr_lginp[idx] + ptr_rnd_normal[idx];
	sup += (wgain * ptr_wmask[i]) * wsup;
	
	float dsup = ptr_dsup[idx];
	ptr_dsup[idx] += (sup - dsup) * taumdt;
}

void update_halfnorm_kernel_cpu(
	int i,
	int dim_z,
	const float *ptr_dsup,
	float* ptr_act,
	float wtagain	
){
	float maxdsup = ptr_dsup[0];
	for(int m=0; m<dim_z; m++){
		int idx = i*dim_z + m;
		float dsup = ptr_dsup[idx];
		if(dsup>maxdsup){
			maxdsup = dsup;
		}
	}
	float maxact = exp(wtagain*maxdsup);
	float vsum = 0;
	for(int m=0; m<dim_z; m++){
		int idx = i*dim_z+m;
		float dsup = ptr_dsup[idx];
		float act = exp(wtagain*(dsup-maxdsup));
		if(maxact<1){
			act *= maxact;
		}
		vsum += act;
		ptr_act[idx] = act;
	}

	if(vsum>1){
		for(int m=0; m<dim_z; m++){
			int idx = i*dim_z + m;
			ptr_act[idx] /= vsum;
		}
	}
}

void update_spkgen_kernel_cpu(
	int i,
	int j,
	int dim_z,
	const float *ptr_act,
	const float* ptr_rnd_uniform01,
	int* ptr_spk,
	float maxfqdt
){
	int idx = i*dim_z+j;
	ptr_spk[idx] = int(ptr_rnd_uniform01[idx]<ptr_act[idx]*maxfqdt);
}



void Group::update_cpu(){
	const int* ptr_conf = static_cast<const int*>(_conf->cpu_data());
	int lginp_idx = ptr_conf[Database::IDX_CONF_STIM];
	int wmask_idx = ptr_conf[Database::IDX_CONF_GAIN_MASK];
	const float* ptr_wmask = _wmask->cpu_data(wmask_idx)+_hcu_start;
	const float* ptr_lginp = _lginp->cpu_data(lginp_idx)+_mcu_start;
	const float* ptr_epsc = _epsc->cpu_data()+(_mcu_start-_mcu_start_in_pop);
	const float* ptr_bj = _bj->cpu_data()+(_mcu_start-_mcu_start_in_pop);
	const float* ptr_rnd_uniform01 = _rnd_uniform01->cpu_data()+_mcu_start;
	const float* ptr_rnd_normal = _rnd_normal->cpu_data()+_mcu_start;
	float* ptr_dsup = _dsup->mutable_cpu_data();
	float* ptr_act = _act->mutable_cpu_data();
	int* ptr_spk = _spike->mutable_cpu_data()+_mcu_start;
	
	int dim_x = _conn_num;
	int dim_y = _hcu_num;
	int dim_z = _mcu_num/_hcu_num; 
	for(int i=0; i<_hcu_num; i++){
		for(int j=0; j<_mcu_num/_hcu_num; j++){
			update_dsup_kernel_cpu(
				i,
				j,
				dim_x,
				dim_y,
				dim_z,
				_mcu_num_in_pop,
				ptr_epsc,
				ptr_bj,
				ptr_lginp,
				ptr_wmask,
				ptr_rnd_normal,
				ptr_dsup,
				_wgain,
				_lgbias,
				_igain,
				_taumdt
			);
		}
		update_halfnorm_kernel_cpu(
			i,
			dim_z,
			ptr_dsup,
			ptr_act,
			_wtagain
		);
		for(int j=0; j<dim_z; j++){
			update_spkgen_kernel_cpu(
				i,
				j,
				dim_z,
				ptr_act,
				ptr_rnd_uniform01,
				ptr_spk,
				_maxfqdt
			);
		}
	}
}

}
}
