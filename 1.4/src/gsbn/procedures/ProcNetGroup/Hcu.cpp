#include "gsbn/procedures/ProcNetGroup/Hcu.hpp"

namespace gsbn{
namespace proc_net_group{

void update_dsup_kernel_cpu(
	int idx,
	int isp_num,
	int mcu_num,
	const float *epsc_epsc,
	const float *ptr_bj,
	const float *ptr_lginp,
	const float *ptr_rnd_normal,
	float *ptr_dsup,
	float wgain,
	float mask,
	float lgbias,
	float igain,
	float taumdt
);
void update_halfnorm_kernel_cpu(
	int idx,
	const float *ptr_dsup,
	float *ptr_act,
	int mcu_num,
	float wtagain
);
void update_spike_kernel_cpu(
	int idx,
	const float *ptr_act,
	const float *ptr_rnd_uniform01,
	int *ptr_spike,
	float maxfqdt
);

/*
Hcu::Hcu(HcuParam hcu_param, Database& db, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){

	CHECK(list_hcu);
	_id = list_hcu->size();
	list_hcu->push_back(this);
	CHECK(_list_hcu = list_hcu);
	CHECK(_list_conn = list_conn);
	CHECK(_msg = msg);

	CHECK(_spike = db.sync_vector_i("spike"));
	CHECK(_rnd_uniform01 = db.sync_vector_f(".rnd_uniform01"));
	CHECK(_rnd_normal = db.sync_vector_f(".rnd_normal"));
	CHECK(_lginp = db.sync_vector_f(".lginp"));
	CHECK(_wmask = db.sync_vector_f(".wmask"));
	CHECK(_conf = db.table("conf"));
	
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt= ptr_conf[Database::IDX_CONF_DT];
	
	int mcu_param_size = hcu_param.mcu_param_size();
	
	_mcu_start = _spike->cpu_vector()->size();
	_slot = new SyncVector<int>();
	_slot->mutable_cpu_vector()->resize(1);
	_slot->mutable_cpu_data()[0]=hcu_param.slot_num();
	db.register_sync_vector_i("hcu_slot_"+to_string(_id), _slot);
	_fanout = new SyncVector<int>();
	db.register_sync_vector_i("mcu_fanout_"+to_string(_id), _fanout);
	
	_mcu_num=0;
	for(int m=0; m<mcu_param_size; m++){
		McuParam mcu_param = hcu_param.mcu_param(m);
		int mcu_num = mcu_param.mcu_num();
		_mcu_num += mcu_num;
		int mcu_fanout = mcu_param.fanout_num();
		for(int n=0; n<mcu_num; n++){
				_spike->mutable_cpu_vector()->push_back(0);
				_fanout->mutable_cpu_vector()->push_back(mcu_fanout);
		}
	}
	
	_dsup = new SyncVector<float>();
	_dsup->mutable_cpu_vector()->resize(_mcu_num);
	db.register_sync_vector_f("dsup_"+to_string(_id), _dsup);
	_act = new SyncVector<float>();
	_act->mutable_cpu_vector()->resize(_mcu_num);
	db.register_sync_vector_f("act_"+to_string(_id), _act);
	
	_epsc = new SyncVector<float>();
	db.register_sync_vector_f("epsc_"+to_string(_id), _epsc);
	_bj = new SyncVector<float>();
	db.register_sync_vector_f("bj_"+to_string(_id), _bj);
	
	_avail_hcu.resize(_mcu_num);
	
	_taumdt = dt/hcu_param.taum();
	_wtagain = hcu_param.wtagain();
	_maxfqdt = hcu_param.maxfq()*dt;
	_igain = hcu_param.igain();
	_wgain = hcu_param.wgain();
	_lgbias = hcu_param.lgbias();
	
}
*/
void Hcu::init_new(HcuParam hcu_param, Database& db, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){

	CHECK(list_hcu);
	_id = list_hcu->size();
	list_hcu->push_back(this);
	CHECK(_list_hcu = list_hcu);
	CHECK(_list_conn = list_conn);
	CHECK(_msg = msg);

	CHECK(_conf = db.table(".conf"));
	CHECK(_spike = db.sync_vector_i("spike"));
	
	int mcu_param_size = hcu_param.mcu_param_size();
	
	CHECK(_slot = db.create_sync_vector_i("hcu_slot_"+to_string(_id)));
	_slot->mutable_cpu_vector()->resize(1, hcu_param.slot_num());
	CHECK(_fanout = db.create_sync_vector_i("mcu_fanout_"+to_string(_id)));
	
	_mcu_num=0;
	int mcu_fanout=0;
	_mcu_start = _spike->cpu_vector()->size();
	for(int m=0; m<mcu_param_size; m++){
		McuParam mcu_param = hcu_param.mcu_param(m);
		int mcu_num = mcu_param.mcu_num();
		mcu_fanout = mcu_param.fanout_num();
		_mcu_num += mcu_num;
		for(int n=0; n<mcu_num; n++){
			_fanout->mutable_cpu_vector()->push_back(mcu_fanout);
		}
		
	}
	_spike->mutable_cpu_vector()->resize(_mcu_start+_mcu_num, 0);
	
	_avail_hcu.resize(_mcu_num);
	
}


void Hcu::init_copy(HcuParam hcu_param, Database& db, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){

	CHECK(list_hcu);
	_id = list_hcu->size();
	list_hcu->push_back(this);
	CHECK(_list_hcu = list_hcu);
	CHECK(_list_conn = list_conn);
	CHECK(_msg = msg);
	
	CHECK(_conf = db.table(".conf"));
	CHECK(_spike = db.sync_vector_i(".fake_spike")); // USE FAKE SPIKE VECTOR TO COUNT MCU NUM
	int mcu_param_size = hcu_param.mcu_param_size();
	
	CHECK(_slot = db.sync_vector_i("hcu_slot_"+to_string(_id)));
	CHECK_EQ(_slot->cpu_vector()->size(), 1);
	
	_mcu_start = _spike->cpu_vector()->size();
	_mcu_num=0;
	for(int m=0; m<mcu_param_size; m++){
		McuParam mcu_param = hcu_param.mcu_param(m);
		int mcu_num = mcu_param.mcu_num();
		_mcu_num += mcu_num;
	}
	_spike->mutable_cpu_vector()->resize(_mcu_start+_mcu_num, 0);
	
	CHECK(_fanout = db.sync_vector_i("mcu_fanout_"+to_string(_id)));
	CHECK_EQ(_fanout->cpu_vector()->size(), _mcu_num);
	
	_avail_hcu.resize(_mcu_num);
	
	CHECK(_spike = db.sync_vector_i("spike"));
}
/*

void Hcu::update_cpu(){
	const int *ptr_conf = static_cast<const int*>(_conf->cpu_data());
	int lginp_idx= ptr_conf[Database::IDX_CONF_STIM];
	int wmask_idx= ptr_conf[Database::IDX_CONF_GAIN_MASK];
	float wmask = (_wmask->cpu_data(wmask_idx))[_id];
	const float* ptr_epsc = _epsc->cpu_data();
	const float* ptr_bj = _bj->cpu_data();
	float *ptr_dsup = _dsup->mutable_cpu_data();
	const float *ptr_lginp = _lginp->cpu_data(lginp_idx)+_mcu_start;
	const float *ptr_rnd_normal = _rnd_normal->cpu_data(_mcu_start);
	for(int idx=0; idx<_mcu_num; idx++){
		update_dsup_kernel_cpu(
			idx,
			_isp.size(),
			_mcu_num,
			ptr_epsc,
			ptr_bj,
			ptr_lginp,
			ptr_rnd_normal,
			ptr_dsup,
			_wgain,
			wmask,
			_lgbias,
			_igain,
			_taumdt
		);
	}
	
	float *ptr_act = _act->mutable_cpu_data();
	for(int idx=0; idx<1; idx++){
		update_halfnorm_kernel_cpu(
			idx,
			ptr_dsup,
			ptr_act,
			_mcu_num,
			_wtagain
		);
	}
	
	int *ptr_spike = _spike->mutable_cpu_data(_mcu_start);
	const float *ptr_rnd_uniform01 = _rnd_uniform01->cpu_data(_mcu_start);
	for(int idx=0; idx<_mcu_num; idx++){
		update_spike_kernel_cpu(
			idx,
			ptr_act,
			ptr_rnd_uniform01,
			ptr_spike,
			_maxfqdt
		);
	}
}
*/

void Hcu::send_receive_cpu(){
	int plasticity = static_cast<const int *>(_conf->cpu_data(0))[Database::IDX_CONF_PLASTICITY];
	if(!plasticity)
		return;
	HOST_VECTOR(int, *v_fanout)=_fanout->mutable_cpu_vector();
	vector<msg_t> list_msg = _msg->receive(_id);
	for(vector<msg_t>::iterator it = list_msg.begin(); it!=list_msg.end(); it++){
		if(it->dest_hcu!=_id){
			continue;
		}
		int isp_size = _isp.size();
		Conn *c;
		switch(it->type){
		case 1:
			if(*(_slot->mutable_cpu_data())>0){
				(*(_slot->mutable_cpu_data()))--;
				_msg->send(it->dest_hcu, it->dest_mcu, it->src_hcu, it->src_mcu, 2);
				for(int i=0; i<isp_size; i++){
					if(it->src_mcu>=_isp_mcu_start[i] && it->src_mcu < _isp_mcu_start[i]+_isp_mcu_num[i]){
						c = _isp[i];
						break;
					}
				}
				c->add_row_cpu(it->src_mcu, it->delay);
				
			}else{
				_msg->send(it->dest_hcu, it->dest_mcu, it->src_hcu, it->src_mcu, 3);
			}
			break;
		case 2:
			break;
		case 3:
			(*v_fanout)[it->dest_mcu - _mcu_start]++;
			_avail_hcu[it->dest_mcu-_mcu_start].push_back(it->src_hcu);
			break;
		default:
			break;
		}
	}
	for(int i=0; i<_mcu_num; i++){
		int mcu_idx = _mcu_start + i;
		CONST_HOST_VECTOR(int, *v_spike)=_spike->cpu_vector();
		if((*v_spike)[mcu_idx]<=0 || (*v_fanout)[i]<=0 || _avail_hcu[i].size()<=0){
			continue;
		}
		(*v_fanout)[i]--;
		float random_number;
		_rnd.gen_uniform01_cpu(&random_number);
		int dest_hcu_idx = ceil(random_number*_avail_hcu[i].size()-1);
		_msg->send(_id, mcu_idx, _avail_hcu[i][dest_hcu_idx], 0, 1);
		_avail_hcu[i].erase(_avail_hcu[i].begin()+dest_hcu_idx);
	}
}

void Hcu::send_receive_gpu(){
	int plasticity = static_cast<const int *>(_conf->cpu_data(0))[Database::IDX_CONF_PLASTICITY];
	if(!plasticity)
		return;

	HOST_VECTOR(int, *v_fanout)=_fanout->mutable_cpu_vector();
	
	vector<msg_t> list_msg = _msg->receive(_id);
	for(vector<msg_t>::iterator it = list_msg.begin(); it!=list_msg.end(); it++){
		if(it->dest_hcu!=_id){
			continue;
		}
		int isp_size = _isp.size();
		Conn *c;
		switch(it->type){
		case 1:
			if(*(_slot->mutable_cpu_data())>0){
				(*(_slot->mutable_cpu_data()))--;
				_msg->send(it->dest_hcu, it->dest_mcu, it->src_hcu, it->src_mcu, 2);
			
				for(int i=0; i<isp_size; i++){
					if(it->src_mcu>=_isp_mcu_start[i] && it->src_mcu < _isp_mcu_start[i]+_isp_mcu_num[i]){
						c = _isp[i];
						break;
					}
				}
				c->add_row_gpu(it->src_mcu, it->delay);
				
			}else{
				_msg->send(it->dest_hcu, it->dest_mcu, it->src_hcu, it->src_mcu, 3);
			}
			break;
		case 2:
			break;
		case 3:
			(*v_fanout)[it->dest_mcu - _mcu_start]++;
			_avail_hcu[it->dest_mcu-_mcu_start].push_back(it->src_hcu);
			break;
		default:
			break;
		}
	}
	

	for(int i=0; i<_mcu_num; i++){
		int mcu_idx = _mcu_start + i;

		CONST_HOST_VECTOR(int, *v_spike)=_spike->cpu_vector();
		if((*v_spike)[mcu_idx]<=0 || (*v_fanout)[i]<=0 || _avail_hcu[i].size()<=0){
			continue;
		}
		(*v_fanout)[i]--;

		float random_number;
		_rnd.gen_uniform01_cpu(&random_number);
		int dest_hcu_idx = ceil(random_number*_avail_hcu[i].size()-1);
		_msg->send(_id, mcu_idx, _avail_hcu[i][dest_hcu_idx], 0, 1);
		_avail_hcu[i].erase(_avail_hcu[i].begin()+dest_hcu_idx);
	}
}

	

////////////////////////////////////////////////////////////////////////////////
// Local functions
////////////////////////////////////////////////////////////////////////////////
/*
void update_dsup_kernel_cpu(
	int idx,
	int isp_num,
	int mcu_num,
	const float *ptr_epsc,
	const float *ptr_bj,
	const float *ptr_lginp,
	const float *ptr_rnd_normal,
	float *ptr_dsup,
	float wgain,
	float wmask,
	float lgbias,
	float igain,
	float taumdt
){
	float wsup=0;
	int offset=0;
	for(int i=0; i<isp_num; i++){
		
		wsup += ptr_bj[idx+offset] + ptr_epsc[idx+offset];
		offset+=mcu_num;
	}
	
	float sup = lgbias + igain * ptr_lginp[idx] + ptr_rnd_normal[idx];
	sup += (wgain * wmask) * wsup;

	float dsup=ptr_dsup[idx];
	ptr_dsup[idx] += (sup - dsup) * taumdt;
}

void update_halfnorm_kernel_cpu(
	int idx,
	const float *ptr_dsup,
	float *ptr_act,
	int mcu_num,
	float wtagain
){
	float maxdsup = ptr_dsup[0];
	for(int i=0; i<mcu_num; i++){
		float dsup=ptr_dsup[i];
		if(dsup>maxdsup){
			maxdsup=dsup;
		}
	}
	float maxact = exp(wtagain*maxdsup);
	float vsum=0;
	for(int i=0; i<mcu_num; i++){
		float dsup = ptr_dsup[i];
		float act = exp(wtagain*(dsup-maxdsup));
		if(maxact<1){
			act *= maxact;
		}
		vsum += act;
		ptr_act[i]=act;
	}
	
	if(vsum>1){
		for(int i=0; i<mcu_num; i++){
			ptr_act[i] /= vsum;
		}
	}
}

void update_spike_kernel_cpu(
	int idx,
	const float *ptr_act,
	const float *ptr_rnd_uniform01,
	int *ptr_spike,
	float maxfqdt
){
	ptr_spike[idx] = int(ptr_rnd_uniform01[idx] < ptr_act[idx]*maxfqdt);
}

*/

}
}
