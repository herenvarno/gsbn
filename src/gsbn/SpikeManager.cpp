#include "gsbn/SpikeManager.hpp"

namespace gsbn{

SpikeManager::SpikeManager(){
}

void SpikeManager::init(Database& db){
	CHECK(_j_array = db.table("j_array"));
	CHECK(_spk = db.table("spk"));
	CHECK(_hcu = db.table("hcu"));
	CHECK(_sup = db.table("sup"));
	CHECK(_stim = db.table("stim"));
	CHECK(_mcu = db.table("mcu"));
	CHECK(_tmp1 = db.table("tmp1"));
	CHECK(_epsc = db.table("epsc"));
	CHECK(_conf = db.table("conf"));
	CHECK(_addr = db.table("addr"));
	
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	
	_kp=ptr_conf[Database::IDX_CONF_KP];
	_ke=ptr_conf[Database::IDX_CONF_KE];
	_kzj=ptr_conf[Database::IDX_CONF_KZJ];
	_kzi=ptr_conf[Database::IDX_CONF_KZI];
	_kftj=ptr_conf[Database::IDX_CONF_KFTJ];
	_bgain=ptr_conf[Database::IDX_CONF_BGAIN];
	_wgain=ptr_conf[Database::IDX_CONF_WGAIN];
	_wtagain=ptr_conf[Database::IDX_CONF_WTAGAIN];
	_igain=ptr_conf[Database::IDX_CONF_IGAIN];
	_eps=ptr_conf[Database::IDX_CONF_EPS];
	_lgbias=ptr_conf[Database::IDX_CONF_LGBIAS];
	_snoise=ptr_conf[Database::IDX_CONF_SNOISE];
	_maxfqdt=ptr_conf[Database::IDX_CONF_MAXFQDT];
	_taumdt=ptr_conf[Database::IDX_CONF_TAUMDT];

}

void SpikeManager::recall(int timestamp){
/*	update_phase_1();
	update_phase_3();
	update_phase_4();
	update_phase_5();
	update_phase_6();*/
}

void SpikeManager::learn(int timestamp, int stim_offset){
	update_phase_1();
	update_phase_2(stim_offset);
	update_phase_3();
	update_phase_4();
	update_phase_5();
	update_phase_6();
}

/*
 * Phase 1: update Pj, Ej, Zj, EPSC
 * No need for timestamp
 */
void update_kernel_phase_1_cpu(
	int idx,
	void *ptr_j_array, int w_j_array,
	void *ptr_epsc, int w_epsc,
	float kzi, float kzj, float ke, float kp){
	
	void *ptr_j_array_data = ptr_j_array+idx*w_j_array;
	float pj = static_cast<float *>(ptr_j_array_data)[Database::IDX_J_ARRAY_PJ];
	float ej = static_cast<float *>(ptr_j_array_data)[Database::IDX_J_ARRAY_EJ];
	float zj = static_cast<float *>(ptr_j_array_data)[Database::IDX_J_ARRAY_ZJ];
	void *ptr_epsc_data = ptr_epsc+idx*w_epsc;
	float epsc = static_cast<float *>(ptr_epsc_data)[Database::IDX_EPSC_VALUE];
	float pdt=0.001;
	pj = (pj - ((ej*kp*kzj - ej*ke*kp + ke*kp*zj)/(ke - kp) +
                    (ke*kp*zj)/(kp - kzj))/(ke - kzj))/exp(kp*pdt) +
        ((exp(kp*pdt - ke*pdt)*(ej*kp*kzj - ej*ke*kp + ke*kp*zj))/(ke - kp) +
         (ke*kp*zj*exp(kp*pdt - kzj*pdt))/(kp - kzj))/(exp(kp*pdt)*(ke - kzj));
	ej = (ej - (ke*zj)/(ke - kzj))/exp(ke*pdt) +
        (ke*zj*exp(ke*pdt - kzj*pdt))/(exp(ke*pdt)*(ke - kzj));
	zj = zj*exp(-kzj*pdt);
	
	static_cast<float *>(ptr_j_array_data)[Database::IDX_J_ARRAY_PJ] = pj;
	static_cast<float *>(ptr_j_array_data)[Database::IDX_J_ARRAY_EJ] = ej;
	static_cast<float *>(ptr_j_array_data)[Database::IDX_J_ARRAY_ZJ] = zj;
	static_cast<float *>(ptr_epsc_data)[Database::IDX_EPSC_VALUE] *= (1 - kzi);
}
void SpikeManager::update_phase_1(){
	int h_j_array =	_j_array->height();
	int w_j_array = _j_array->width();
	void *ptr_j_array = _j_array->mutable_cpu_data();
	int w_epsc = _epsc->width();
	void *ptr_epsc = _epsc->mutable_cpu_data();
	for(int i=0;i<h_j_array;i++){
		update_kernel_phase_1_cpu(
			i,
			ptr_j_array, w_j_array,
			ptr_epsc, w_epsc,
			_kzi, _kzj, _ke, _kp);
	}
}

/*
 * Phase 2: update dsup
 * No need for timestamp
 */
void update_kernel_phase_2_cpu(
	int idx,
	const void *ptr_mcu, int w_mcu,
	const void *ptr_epsc, int w_epsc,
	const void *ptr_j_array, int w_j_array,
	const void *ptr_stim, int w_stim,
	void *ptr_sup, int w_sup,
	float lgbias, float igain, float snoise, float taumdt,
	float wgain, float bgain,
	float eps){

	const int *ptr_mcu_data = static_cast<const int *>(ptr_mcu+idx*w_mcu);
	int j_array_idx = ptr_mcu_data[Database::IDX_MCU_J_ARRAY_INDEX];
	int j_array_num = ptr_mcu_data[Database::IDX_MCU_J_ARRAY_NUM];
	
	float *ptr_sup_data = static_cast<float *>(ptr_sup+idx*w_sup);
	float dsup = ptr_sup_data[Database::IDX_SUP_DSUP];
	float wsup=0;
	for(int i=0; i<j_array_num; i++){
		const float *ptr_j_array_data=static_cast<const float *>(ptr_j_array+(i+j_array_idx)*w_j_array);
		float pj=ptr_j_array_data[Database::IDX_J_ARRAY_PJ];
		const float *ptr_epsc_data=static_cast<const float *>(ptr_epsc+(i+j_array_idx)*w_epsc);
		float epsc=ptr_epsc_data[Database::IDX_EPSC_VALUE];
		float bj = bgain * log(pj + eps);
		wsup += bj + epsc;
	}
	const float *ptr_stim_data=static_cast<const float *>(ptr_stim+idx*w_stim); //FIXME : CHECK INDEX OF STIM.
	float sup = lgbias + igain * ptr_stim_data[0] + Random::gen_normal(0, snoise);
	sup += wgain * wsup;
	dsup += (sup - dsup) * taumdt;
	ptr_sup_data[Database::IDX_SUP_DSUP]=dsup;
}
void SpikeManager::update_phase_2(int stim_offset){
	int h_sup=_sup->height();
	int w_sup=_sup->width();
	void *ptr_sup=_sup->mutable_cpu_data();
	int w_mcu = _mcu->width();
	const void *ptr_mcu = _mcu->cpu_data();
	int w_epsc = _epsc->width();
	const void *ptr_epsc = _epsc->cpu_data();
	int w_j_array = _j_array->width();
	const void *ptr_j_array = _j_array->cpu_data();
	int w_stim = _stim->width();
	const void *ptr_stim = _stim->cpu_data(stim_offset);

	for(int i=0; i<h_sup; i++){
		update_kernel_phase_2_cpu(
			i,
			ptr_mcu, w_mcu,
			ptr_epsc, w_epsc,
			ptr_j_array, w_j_array,
			ptr_stim, w_stim,
			ptr_sup, w_sup,
			_lgbias, _igain, _snoise, _taumdt,
			_wgain, _bgain,
			_eps);
	}
}

/*
 * Phase 3: halfnormlize
 * No need for timestamp
 */
void update_kernel_phase_3_cpu(
	int idx, 
	const void *ptr_hcu, int w_hcu,
	void *ptr_sup, int w_sup,
	float wtagain){

	const int *ptr_hcu_data = static_cast<const int *>(ptr_hcu+idx*w_hcu);
	int mcu_idx = ptr_hcu_data[Database::IDX_HCU_MCU_INDEX];
	int mcu_num = ptr_hcu_data[Database::IDX_HCU_MCU_NUM];
	
	float maxdsup = static_cast<float *>(ptr_sup+mcu_idx*w_sup)[Database::IDX_SUP_DSUP];
	for(int i=0; i<mcu_num; i++){
		float dsup=static_cast<float *>(ptr_sup+(mcu_idx+i)*w_sup)[Database::IDX_SUP_DSUP];
		if(dsup>maxdsup){
			maxdsup=dsup;
		}
	}
	float maxact = exp(wtagain*maxdsup);
	float vsum=0;
	for(int i=0; i<mcu_num; i++){
		float *ptr_sup_data = static_cast<float *>(ptr_sup+(mcu_idx+i)*w_sup);
		float dsup = ptr_sup_data[Database::IDX_SUP_DSUP];
		float act = exp(wtagain*(dsup-maxdsup));
		act = maxact<1?act*maxact:act;
		vsum += act;
		ptr_sup_data[Database::IDX_SUP_ACT]=act;
	}
	
	if(vsum>1){
		for(int i=0; i<mcu_num; i++){
			static_cast<float *>(ptr_sup+(mcu_idx+i)*w_sup)[Database::IDX_SUP_ACT]/=vsum;
		}
	}
}
void SpikeManager::update_phase_3(){
	int h_hcu=_hcu->height();
	int w_hcu=_hcu->width();
	const void *ptr_hcu=_hcu->cpu_data();
	int w_sup=_sup->width();
	void *ptr_sup=_sup->mutable_cpu_data();
	
	for(int i=0; i<h_hcu; i++){
		update_kernel_phase_3_cpu(
			i,
			ptr_hcu, w_hcu,
			ptr_sup, w_sup,
			_wtagain);
	}
}

/*
 * Phase 4: generate spike
 * No need for timestamp
 */
void update_kernel_phase_4_cpu(
	int idx,
	const void *ptr_sup, int w_sup,
	void *ptr_spk, int w_spk,
	float maxfqdt){

	unsigned char *ptr_spk_data = static_cast<unsigned char *>(ptr_spk+idx*w_spk);
	float act = static_cast<const float *>(ptr_sup+idx*w_sup)[Database::IDX_SUP_ACT];
	ptr_spk_data[Database::IDX_SPK_VALUE] = Random::gen_uniform01()<act*maxfqdt;
}
void SpikeManager::update_phase_4(){
	int h_spk=_spk->height();
	int w_spk=_spk->width();
	void *ptr_spk=_spk->mutable_cpu_data();
	int w_sup=_sup->width();
	const void *ptr_sup=_sup->cpu_data(); 
	for(int i=0; i<h_spk; i++){
		update_kernel_phase_4_cpu(
			i,
			ptr_sup, w_sup,
			ptr_spk, w_spk,
			_maxfqdt);
	}
}

/*
 * Phase 5: generate short spike list, tmp table
 * No need for timestamp
 */
void SpikeManager::update_phase_5(){
	int h_spk=_spk->height();
	_tmp1->reset();
	for(int i=0; i<h_spk; i++){
		const unsigned char spike = static_cast<const unsigned char *>(_spk->cpu_data(i, 0))[0];
		if(spike){
			int *ptr_tmp1 = static_cast<int *>(_tmp1->expand(1));
			ptr_tmp1[0]=i;
		}
	}
}

/*
 * Phase 6: increase Zj
 * No need for timestamp
 */
void update_kernel_phase_6_cpu(
	int idx_tmp1, 
	const void *ptr_tmp1, int w_tmp1,
	const void *ptr_mcu, int w_mcu,
	void *ptr_j_array, int w_j_array,
	float kftj){

	int idx_mcu = static_cast<const int *>(ptr_tmp1+idx_tmp1*w_tmp1)[0];
	const int *ptr_mcu_data = static_cast<const int *>(ptr_mcu+idx_mcu*w_mcu);
	int idx_j_array = ptr_mcu_data[0];
	int num_j_array = ptr_mcu_data[1];
	for(int i=0; i<num_j_array; i++){
		float *ptr_j_array_data = static_cast<float *>(ptr_j_array+(idx_j_array+i)*w_j_array);
		ptr_j_array_data[2]+=kftj;
	}
}
void SpikeManager::update_phase_6(){
	int h_tmp1=_tmp1->height();
	int w_tmp1=_tmp1->width();
	if(h_tmp1<=0)
		return;
	const void *ptr_tmp1=_tmp1->cpu_data();
	int w_mcu=_mcu->width();
	const void *ptr_mcu=_mcu->cpu_data();
	int w_j_array = _j_array->width();
	void * ptr_j_array = _j_array->mutable_cpu_data();
	for(int i=0; i<h_tmp1; i++){
		update_kernel_phase_6_cpu(
			i,
			ptr_tmp1, w_tmp1,
			ptr_mcu, w_mcu,
			ptr_j_array, w_j_array,
			_kftj);
	}
}


}
