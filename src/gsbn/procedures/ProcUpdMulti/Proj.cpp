#include "gsbn/procedures/ProcUpdMulti/Proj.hpp"

namespace gsbn{
namespace proc_upd_multi{

void Proj::init_new(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop){
	
	CHECK(list_pop);
	CHECK(list_proj);
	_list_pop = list_pop;
	_list_proj = list_proj;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	int slot_num=proj_param.slot_num();
	if(_dim_conn > slot_num){
		_dim_conn = slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;

	_device = _ptr_dest_pop->_device;
	_rank = _ptr_dest_pop->_rank;
	
	float dt;
	int rank;
	CHECK(_glv.getf("dt", dt));
	CHECK(_glv.geti("rank", rank));

	_tauzidt=dt/proj_param.tauzi();
	_tauzjdt=dt/proj_param.tauzj();
	_tauedt=dt/proj_param.taue();
	_taupdt=dt/proj_param.taup();
	_eps=dt/proj_param.taup();
	_eps2=(dt/proj_param.taup())*(dt/proj_param.taup());
	_kfti=1/(proj_param.maxfq() * proj_param.tauzi());
	_kftj=1/(proj_param.maxfq() * proj_param.tauzj());
	_bgain=proj_param.bgain();
	_wgain=proj_param.wgain();
	if(proj_param.has_tauepsc()){
		_tauepscdt = dt/proj_param.tauepsc();
	}else{
		_tauepscdt = _tauzidt;
	}
	
	if(_rank != rank){
		return;
	}
	
	p->_epsc->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	p->_bj->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;
	
	CHECK(_ii = db.create_sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.create_sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.create_sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_siq = db.create_sync_vector_i32(".siq_"+to_string(_id)));
	CHECK(_pi = db.create_sync_vector_f32("pi_"+to_string(_id)));
	CHECK(_ei = db.create_sync_vector_f32("ei_"+to_string(_id)));
	CHECK(_zi = db.create_sync_vector_f32("zi_"+to_string(_id)));
	CHECK(_ti = db.create_sync_vector_i32("ti_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i32(".ssj_"+to_string(_id)));
	CHECK(_pj = db.create_sync_vector_f32("pj_"+to_string(_id)));
	CHECK(_ej = db.create_sync_vector_f32("ej_"+to_string(_id)));
	CHECK(_zj = db.create_sync_vector_f32("zj_"+to_string(_id)));
	CHECK(_pij = db.create_sync_vector_f32("pij_"+to_string(_id)));
	CHECK(_eij = db.create_sync_vector_f32("eij_"+to_string(_id)));
	CHECK(_zj2 = db.create_sync_vector_f32("zj2_"+to_string(_id)));
	CHECK(_wij = db.create_sync_vector_f32("wij_"+to_string(_id)));
	CHECK(_si = db.create_sync_vector_i8("si_"+to_string(_id)));
	
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	Parser par(proc_param);
	if(!par.argi("spike buffer size", _spike_buffer_size)){
		_spike_buffer_size = 1;
	}else{
		CHECK_GT(_spike_buffer_size, 0);
	}
	_ii->resize(_dim_hcu * _dim_conn, -1);
	_qi->resize(_dim_hcu * _dim_conn);
	_di->resize(_dim_hcu * _dim_conn);
	_pi->resize(_dim_hcu * _dim_conn, 1.0/_dim_conn);
	_ei->resize(_dim_hcu * _dim_conn);
	_zi->resize(_dim_hcu * _dim_conn);
	_ti->resize(_dim_hcu * _dim_conn);
	_siq->resize(_dim_hcu * _dim_conn);
	_pj->resize(_dim_hcu * _dim_mcu, 1.0/_dim_mcu);
	_ej->resize(_dim_hcu * _dim_mcu);
	_zj->resize(_dim_hcu * _dim_mcu);
	_pij->resize(_dim_hcu * _dim_conn * _dim_mcu, 1.0/_dim_conn/_dim_mcu);
	_eij->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zj2->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_wij->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_si->resize(_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
}

void Proj::init_copy(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop){
	
	CHECK(list_pop);
	CHECK(list_proj);
	_list_pop = list_pop;
	_list_proj = list_proj;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	int slot_num=proj_param.slot_num();
	if(_dim_conn > slot_num){
		_dim_conn = slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;

	_device = _ptr_dest_pop->_device;
	_rank = _ptr_dest_pop->_rank;
	
	float dt;
	int rank;
	CHECK(_glv.getf("dt", dt));
	CHECK(_glv.geti("rank", rank));
	
	_tauzidt=dt/proj_param.tauzi();
	_tauzjdt=dt/proj_param.tauzj();
	_tauedt=dt/proj_param.taue();
	_taupdt=dt/proj_param.taup();
	_eps=dt/proj_param.taup();
	_eps2=(dt/proj_param.taup())*(dt/proj_param.taup());
	_kfti=1/(proj_param.maxfq() * proj_param.tauzi());
	_kftj=1/(proj_param.maxfq() * proj_param.tauzj());
	_bgain=proj_param.bgain();
	_wgain=proj_param.wgain();
	if(proj_param.has_tauepsc()){
		_tauepscdt = dt/proj_param.tauepsc();
	}else{
		_tauepscdt = _tauzidt;
	}

	if(_rank != rank){
		return;
	}

	p->_epsc->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	p->_bj->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;
	CHECK(_ii = db.sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_siq = db.create_sync_vector_i32(".siq_"+to_string(_id)));
	CHECK(_pi = db.sync_vector_f32("pi_"+to_string(_id)));
	CHECK(_ei = db.sync_vector_f32("ei_"+to_string(_id)));
	CHECK(_zi = db.sync_vector_f32("zi_"+to_string(_id)));
	CHECK(_ti = db.sync_vector_i32("ti_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i32(".ssj_"+to_string(_id)));
	CHECK(_pj = db.sync_vector_f32("pj_"+to_string(_id)));
	CHECK(_ej = db.sync_vector_f32("ej_"+to_string(_id)));
	CHECK(_zj = db.sync_vector_f32("zj_"+to_string(_id)));
	CHECK(_pij = db.sync_vector_f32("pij_"+to_string(_id)));
	CHECK(_eij = db.sync_vector_f32("eij_"+to_string(_id)));
	CHECK(_zj2 = db.sync_vector_f32("zj2_"+to_string(_id)));
	CHECK(_wij = db.sync_vector_f32("wij_"+to_string(_id)));
	CHECK(_si = db.sync_vector_i8("si_"+to_string(_id)));
	
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	Parser par(proc_param);
	if(!par.argi("spike buffer size", _spike_buffer_size)){
		_spike_buffer_size = 1;
	}else{
		CHECK_GT(_spike_buffer_size, 0);
	}
	
	CHECK_EQ(_ii->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_qi->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_di->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_pi->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ei->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_zi->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ti->size(), _dim_hcu * _dim_conn);
	_siq->resize(_dim_hcu * _dim_conn);
	CHECK_EQ(_pj->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_ej->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_zj->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_pij->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_eij->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_zj2->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_wij->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_si->size(), _ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
}

void update_all_kernel_cpu(
	int i,
	int j,
	int spike_buffer_size,
	int dim_conn,
	int dim_hcu,
	int dim_mcu,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const int8_t *ptr_sj,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zj2,
	float *ptr_wij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kftj,
	float wgain,
	float eps,
	float eps2,
	int *active_flag
){
	float shared_zi = ptr_zi[i];
	int shared_ti = ptr_ti[i];
	if(j==0){
		float pi = ptr_pi[i];
		float zi = shared_zi;
		int ti = shared_ti;
		int pdt = simstep - ti;
		if(pdt>0){
			float ei = ptr_ei[i];
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			ptr_pi[i] = pi;
			ptr_ei[i] = ei;
		}
	}
	
	int index = i*dim_mcu+j;

	float pij = ptr_pij[index];
	float eij = ptr_eij[index];
	float zj2 = ptr_zj2[index];
	
	float zi2 = shared_zi;
	int tij = shared_ti;
	int simstep_now = tij;
	if(simstep - simstep_now>=spike_buffer_size){
		simstep_now = simstep - spike_buffer_size +1;
	}
	
	while(simstep_now < simstep){
		simstep_now++;
		int idx_sj = (simstep_now%spike_buffer_size)*dim_hcu*dim_mcu+(i/dim_conn)*dim_mcu+j;
		if(ptr_sj[idx_sj]>0){
			int pdt = simstep_now - tij;
			if(pdt>0){
				pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
						(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
						((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
						(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
						(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
				eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
						(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
				zi2 = zi2*exp(-kzi*pdt);
				zj2 = zj2*exp(-kzj*pdt)+kftj;
				tij = simstep_now;
			}
		}
	}
	int pdt = simstep - tij;
	if(pdt>0){
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
			(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
			(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zj2 = zj2*exp(-kzj*pdt);
	}
	ptr_pij[index] = pij;
	ptr_eij[index] = eij;
	ptr_zj2[index] = zj2;
			
		// update wij and epsc
		float wij;
		if(kp){
			float pi = ptr_pi[i];
			float pj = ptr_pj[i/dim_conn*dim_mcu + j];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
			if(wij >-1000){
				*active_flag = 1;
			}
		}else{
			*active_flag = 1;
		}
}


void update_jxx_kernel_cpu(
	int idx,
	const int8_t *ptr_sj,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_bj,
	float *ptr_epsc,
	float kp,
	float ke,
	float kzj,
	float kzi,
	float kepsc,
	float kftj,
	float bgain,
	float eps
){
	float pj = ptr_pj[idx];
	float ej = ptr_ej[idx];
	float zj = ptr_zj[idx];
	int sj = ptr_sj[idx];
	
	ptr_epsc[idx] *= (1-kepsc);
	
	pj += (ej - pj)*kp;
	ej += (zj - ej)*ke;
	zj *= (1-kzj);
	
	if(sj>0){
		zj += kftj;
	}
	
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj[idx] = bj;
	}
	
	ptr_pj[idx] = pj;
	ptr_ej[idx] = ej;
	ptr_zj[idx] = zj;
}


void update_row_kernel_cpu(
	int i,
	int j,
	int spike_buffer_size,
	int dim_conn,
	int dim_hcu,
	int dim_mcu,
	const int *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const int8_t *ptr_sj,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zj2,
	float* ptr_wij,
	float* ptr_epsc,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfti,
	float kftj,
	float wgain,
	float eps,
	float eps2,
	int *active_flag
){
	int row = ptr_ssi[i];
	int col = j;
	int index = row*dim_mcu+col;
	
	float shared_zi = ptr_zi[row];
	int shared_ti = ptr_ti[row];
	if(j==0){
		float pi = ptr_pi[row];
		float ei = ptr_ei[row];
		float zi = shared_zi;
		int ti = shared_ti;
		int pdt = simstep - ti;
		if(pdt>0){
			float pi = ptr_pi[row];
			float ei = ptr_ei[row];
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			ptr_pi[row] = pi;
			ptr_ei[row] = ei;
		}
	}
	
	float pij = ptr_pij[index];
	float eij = ptr_eij[index];
	float zj2 = ptr_zj2[index];

	float zi2 = shared_zi;
	int tij = shared_ti;
	int simstep_now = tij;

	if(simstep - simstep_now>=spike_buffer_size){
		simstep_now = simstep - spike_buffer_size +1;
	}
	
	while(simstep_now < simstep){
		simstep_now++;
		int idx_sj = (simstep_now%spike_buffer_size)*dim_hcu*dim_mcu+(row/dim_conn)*dim_mcu+j;
		if(ptr_sj[idx_sj]>0 && simstep_now<simstep){
			int pdt = simstep_now - tij;
			if(pdt>0){
				pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
						(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
						((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
						(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
						(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
				eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
						(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
				zi2 = zi2*exp(-kzi*pdt);
				zj2 = zj2*exp(-kzj*pdt)+kftj;
				tij = simstep_now;
			}
		}
	}
	int pdt = simstep - tij;
	pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
		(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
		((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
		(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
		(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
	eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
		(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
	zj2 = zj2*exp(-kzj*pdt);
	
	int idx_sj = (simstep%spike_buffer_size)*dim_hcu*dim_mcu+(row/dim_conn)*dim_mcu+j;
	if(ptr_sj[idx_sj]>0){
		zj2 += kftj;
	}
	
	ptr_pij[index] = pij;
	ptr_eij[index] = eij;
	ptr_zj2[index] = zj2;
		
	float wij;
	int idx_hcu = row / dim_conn;
	int idx_mcu = idx_hcu * dim_mcu + j;
	if(kp){
		float pi = ptr_pi[row];
		float pj = ptr_pj[idx_mcu];
		wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
		ptr_wij[index] = wij;
		if(wij >-1000){
			*active_flag = 1;
		}
	}else{
		wij = ptr_wij[index];
		*active_flag = 1;
	}
	
	ptr_epsc[idx_mcu] += wij;
}

void Proj::update_all_cpu(){
	int simstep;
	float prn;
	float old_prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("prn", prn));
	CHECK(_glv.getf("old-prn", old_prn));
	if(old_prn!=prn){
		float *ptr_pi = _pi->mutable_cpu_data();
		float *ptr_ei = _ei->mutable_cpu_data();
		float *ptr_zi = _zi->mutable_cpu_data();
		int *ptr_ti = _ti->mutable_cpu_data();
		const float *ptr_pj = _pj->cpu_data();
		float *ptr_pij = _pij->mutable_cpu_data();
		float *ptr_eij = _eij->mutable_cpu_data();
		float *ptr_zj2 = _zj2->mutable_cpu_data();
		float *ptr_wij = _wij->mutable_cpu_data();
		const int8_t *ptr_sj = _sj->cpu_data();

		for(int i=0; i<_dim_hcu * _dim_conn; i++){
			int active_flag=0;
			for(int j=0; j<_dim_mcu; j++){
				update_all_kernel_cpu(
					i,
					j,
					_spike_buffer_size,
					_dim_conn,
					_dim_hcu,
					_dim_mcu,
					ptr_pi,
					ptr_ei,
					ptr_zi,
					ptr_ti,
					ptr_sj,
					ptr_pj,
					ptr_pij,
					ptr_eij,
					ptr_zj2,
					ptr_wij,
					simstep-1,
					_taupdt*old_prn,
					_tauedt,
					_tauzidt,
					_tauzjdt,
					_kftj,
					_wgain,
					_eps,
					_eps2,
					&active_flag
				);
			}
			int ti = ptr_ti[i];
			int pdt = simstep - 1 - ti;
			if(pdt>0){
				float zi = ptr_zi[i];
				zi = zi*exp(-_tauzidt*pdt);
				ptr_zi[i] = zi;
			}
			ptr_ti[i] = simstep-1;
			
			if(_taupdt*old_prn){
				if(!active_flag){
					ptr_ti[i] = -1;
				}
			}
		}
	}
}

void Proj::update_jxx_cpu(){
	int simstep;
	float prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("prn", prn));
	float *ptr_pj = _pj->mutable_cpu_data();
	float *ptr_ej = _ej->mutable_cpu_data();
	float *ptr_zj = _zj->mutable_cpu_data();
	float *ptr_epsc = _epsc->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_bj = _bj->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int8_t *ptr_sj = _sj->cpu_data()+(simstep%_spike_buffer_size)*_dim_hcu*_dim_mcu;

	for(int i=0; i<_dim_hcu * _dim_mcu; i++){
		update_jxx_kernel_cpu(
			i,
			ptr_sj,
			ptr_pj,
			ptr_ej,
			ptr_zj,
			ptr_bj,
			ptr_epsc,
			_taupdt*prn,
			_tauedt,
			_tauzjdt,
			_tauzidt,
			_tauepscdt,
			_kftj,
			_bgain,
			_eps
		);
	}
}

void Proj::update_ssi_cpu(){
	CONST_HOST_VECTOR(int, *v_siq) = _siq->cpu_vector();
	HOST_VECTOR(int, *v_ssi) = _ssi->mutable_cpu_vector();
	v_ssi->clear();
	for(int i=0; i<_dim_conn * _dim_hcu; i++){
		if((*v_siq)[i]>0){
			v_ssi->push_back(i);
		}
	}
}

void Proj::update_ssj_cpu(){
	int simstep;
	CHECK(_glv.geti("simstep", simstep));
	CONST_HOST_VECTOR(int8_t, *v_sj) = _sj->host_vector(0);
	HOST_VECTOR(int, *v_ssj) = _ssj->mutable_host_vector(0);
	int offset=(simstep%_spike_buffer_size)*_dim_hcu*_dim_mcu;
	v_ssj->clear();
	for(int i=0; i<_dim_hcu * _dim_hcu; i++){
		if((*v_sj)[i+offset]>0){
			v_ssj->push_back(i);
		}
	}
}

void Proj::update_que_cpu(){
	CONST_HOST_VECTOR(int, *v_ii) = _ii->cpu_vector();
	CONST_HOST_VECTOR(int, *v_di) = _di->cpu_vector();
	CONST_HOST_VECTOR(int8_t, *v_si) = _si->cpu_vector();
	HOST_VECTOR(int, *v_qi) = _qi->mutable_cpu_vector();
	HOST_VECTOR(int, *v_siq) = _siq->mutable_cpu_vector();
	
	for(int i=0; i<_dim_conn * _dim_hcu; i++){
		if((*v_ii)[i]<0){
			continue;
		}
		(*v_siq)[i] = (*v_qi)[i] & 0x01;
		(*v_qi)[i] >>= 1;
		
		int8_t spk = (*v_si)[(*v_ii)[i]];
		if(spk>0){
			(*v_qi)[i] |= (0x01 << ((*v_di)[i]-1));
		}
	}
}

void Proj::update_row_cpu(){
	int simstep;
	float prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("prn", prn));
	
	float *ptr_pi = _pi->mutable_cpu_data();
	float *ptr_ei = _ei->mutable_cpu_data();
	float *ptr_zi = _zi->mutable_cpu_data();
	int *ptr_ti = _ti->mutable_cpu_data();
	const float *ptr_pj = _pj->cpu_data();
	float *ptr_pij = _pij->mutable_cpu_data();
	float *ptr_eij = _eij->mutable_cpu_data();
	float *ptr_zj2 = _zj2->mutable_cpu_data();
	float *ptr_wij = _wij->mutable_cpu_data();
	float *ptr_epsc = _epsc->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	int8_t *ptr_sj = _sj->mutable_cpu_data();
	
	const int *ptr_ssi = _ssi->cpu_data();
	const int *ptr_ii = _ii->cpu_data();
	int active_row_num = _ssi->size();
	for(int i=0; i<active_row_num; i++){
		int active_flag=0;
		for(int j=0; j<_dim_mcu; j++){
			update_row_kernel_cpu(
				i,
				j,
				_spike_buffer_size,
				_dim_conn,
				_dim_hcu,
				_dim_mcu,
				ptr_ssi,
				ptr_pi,
				ptr_ei,
				ptr_zi,
				ptr_ti,
				ptr_sj,
				ptr_pj,
				ptr_pij,
				ptr_eij,
				ptr_zj2,
				ptr_wij,
				ptr_epsc,
				simstep,
				_taupdt*prn,
				_tauedt,
				_tauzidt,
				_tauzjdt,
				_kfti,
				_kftj,
				_wgain,
				_eps,
				_eps2,
				&active_flag
			);
		}
		int ti = ptr_ti[ptr_ssi[i]];
		int pdt = simstep - ti;
		if(pdt>0){
			float zi = ptr_zi[ptr_ssi[i]];
			zi = zi*exp(-_tauzidt*pdt)+_kfti;
			ptr_zi[ptr_ssi[i]] = zi;
		}
		ptr_ti[ptr_ssi[i]] = simstep;
		
		if(_taupdt*prn){
			if(!active_flag){
				ptr_ti[ptr_ssi[i]] = -1;
			}
		}
	}
}

}
}
