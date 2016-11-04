#include "gsbn/procedures/ProcNet/Conn.hpp"

namespace gsbn{
namespace proc_net{

void full_update_i_kernel_cpu(
	int idx,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	int simstep,
	float kp,
	float ke,
	float kzi
);
void full_update_ij_kernel_cpu(
	int idx,
	int w,
	const float *ptr_pi,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float *ptr_wij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float wgain,
	float eps,
	float eps2
);


void update_i_kernel_cpu(
	int idx,
	const int *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kfti
);

void update_j_kernel_cpu(
	int idx,
	const int *ptr_sj,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_bj,
	float *ptr_epsc,
	float kp,
	float ke,
	float kzj,
	float kzi,
	float kftj,
	float bgain,
	float eps
);

void update_ij_row_kernel_cpu(
	int idx,
	int w,
	const int *ptr_ssi,
	const float *ptr_pi,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float *ptr_wij,
	float *ptr_epsc,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfti,
	float wgain,
	float eps,
	float eps2
);

void update_ij_col_kernel_cpu(
	int idx,
	int h,
	int w,
	const int *ptr_ssj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kftj
);



void Conn::init_new(ProjParam proj_param, Database& db, vector<Conn*>* list_conn, int w){
	CHECK(list_conn);
	_id=list_conn->size();
	list_conn->push_back(this);
	
	CHECK_GT(w, 0);
	_w=w;
	_h=0;
	
	CHECK(_ii = db.create_sync_vector_i("ii_"+to_string(_id)));
	CHECK(_qi = db.create_sync_vector_i("qi_"+to_string(_id)));
	CHECK(_di = db.create_sync_vector_i("di_"+to_string(_id)));
	CHECK(_si = db.create_sync_vector_i("si_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i(".ssi_"+to_string(_id)));
	CHECK(_pi = db.create_sync_vector_f("pi_"+to_string(_id)));
	CHECK(_ei = db.create_sync_vector_f("ei_"+to_string(_id)));
	CHECK(_zi = db.create_sync_vector_f("zi_"+to_string(_id)));
	CHECK(_ti = db.create_sync_vector_i("ti_"+to_string(_id)));
	CHECK(_sj = db.create_sync_vector_i("sj_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i(".ssj_"+to_string(_id)));
	CHECK(_pj = db.create_sync_vector_f("pj_"+to_string(_id)));
	CHECK(_ej = db.create_sync_vector_f("ej_"+to_string(_id)));
	CHECK(_zj = db.create_sync_vector_f("zj_"+to_string(_id)));
	CHECK(_pij = db.create_sync_vector_f("pij_"+to_string(_id)));
	CHECK(_eij = db.create_sync_vector_f("eij_"+to_string(_id)));
	CHECK(_zi2 = db.create_sync_vector_f("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.create_sync_vector_f("zj2_"+to_string(_id)));
	CHECK(_tij = db.create_sync_vector_i("tij_"+to_string(_id)));
	CHECK(_wij = db.create_sync_vector_f("wij_"+to_string(_id)));
	CHECK(_spike = db.sync_vector_i("spike"));
	
	#ifndef CPU_ONLY
	_sj->mutable_cpu_vector()->resize(w);
	thrust::fill(_sj->mutable_cpu_vector()->begin(), _sj->mutable_cpu_vector()->end(), 0);
	_pj->mutable_cpu_vector()->resize(w);
	thrust::fill(_pj->mutable_cpu_vector()->begin(), _pj->mutable_cpu_vector()->end(), 1.0/w);
	_ej->mutable_cpu_vector()->resize(w);
	thrust::fill(_ej->mutable_cpu_vector()->begin(), _ej->mutable_cpu_vector()->end(), 0);
	_zj->mutable_cpu_vector()->resize(w);
	thrust::fill(_zj->mutable_cpu_vector()->begin(), _zj->mutable_cpu_vector()->end(), 0);
	#else
	_sj->mutable_cpu_vector()->resize(w);
	std::fill(_sj->mutable_cpu_vector()->begin(), _sj->mutable_cpu_vector()->end(), 0);
	_pj->mutable_cpu_vector()->resize(w);
	std::fill(_pj->mutable_cpu_vector()->begin(), _pj->mutable_cpu_vector()->end(), 1.0/w);
	_ej->mutable_cpu_vector()->resize(w);
	std::fill(_ej->mutable_cpu_vector()->begin(), _ej->mutable_cpu_vector()->end(), 0);
	_zj->mutable_cpu_vector()->resize(w);
	std::fill(_zj->mutable_cpu_vector()->begin(), _zj->mutable_cpu_vector()->end(), 0);
	#endif

	CHECK(_conf = db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt= ptr_conf[Database::IDX_CONF_DT];
	
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
	_pi0=proj_param.pi0();
}

void Conn::init_copy(ProjParam proj_param, Database& db, vector<Conn*>* list_conn, int w){
	CHECK(list_conn);
	_id=list_conn->size();
	list_conn->push_back(this);
	
	CHECK_GT(w, 0);
	_w=w;
	
	CHECK(_ii = db.sync_vector_i("ii_"+to_string(_id)));
	CHECK(_qi = db.sync_vector_i("qi_"+to_string(_id)));
	CHECK(_di = db.sync_vector_i("di_"+to_string(_id)));
	CHECK(_si = db.sync_vector_i("si_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i(".ssi_"+to_string(_id)));
	CHECK(_pi = db.sync_vector_f("pi_"+to_string(_id)));
	CHECK(_ei = db.sync_vector_f("ei_"+to_string(_id)));
	CHECK(_zi = db.sync_vector_f("zi_"+to_string(_id)));
	CHECK(_ti = db.sync_vector_i("ti_"+to_string(_id)));
	CHECK(_sj = db.sync_vector_i("sj_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i(".ssj_"+to_string(_id)));
	CHECK(_pj = db.sync_vector_f("pj_"+to_string(_id)));
	CHECK(_ej = db.sync_vector_f("ej_"+to_string(_id)));
	CHECK(_zj = db.sync_vector_f("zj_"+to_string(_id)));
	CHECK(_pij = db.sync_vector_f("pij_"+to_string(_id)));
	CHECK(_eij = db.sync_vector_f("eij_"+to_string(_id)));
	CHECK(_zi2 = db.sync_vector_f("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.sync_vector_f("zj2_"+to_string(_id)));
	CHECK(_tij = db.sync_vector_i("tij_"+to_string(_id)));
	CHECK(_wij = db.sync_vector_f("wij_"+to_string(_id)));
	CHECK(_spike = db.sync_vector_i("spike"));
	
	CHECK_EQ(_w, _sj->cpu_vector()->size());
	CHECK_EQ(_w, _pj->cpu_vector()->size());
	CHECK_EQ(_w, _ej->cpu_vector()->size());
	CHECK_EQ(_w, _zj->cpu_vector()->size());
	_h = _ii->cpu_vector()->size();
	CHECK_EQ(_h, _qi->cpu_vector()->size());
	CHECK_EQ(_h, _di->cpu_vector()->size());
	CHECK_EQ(_h, _si->cpu_vector()->size());
	CHECK_EQ(_h, _pi->cpu_vector()->size());
	CHECK_EQ(_h, _ei->cpu_vector()->size());
	CHECK_EQ(_h, _zi->cpu_vector()->size());
	CHECK_EQ(_h, _ti->cpu_vector()->size());
	CHECK_EQ(_h*_w, _pij->cpu_vector()->size());
	CHECK_EQ(_h*_w, _eij->cpu_vector()->size());
	CHECK_EQ(_h*_w, _zi2->cpu_vector()->size());
	CHECK_EQ(_h*_w, _zj2->cpu_vector()->size());
	CHECK_EQ(_h*_w, _tij->cpu_vector()->size());
	CHECK_EQ(_h*_w, _wij->cpu_vector()->size());
	
	CHECK(_conf = db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt= ptr_conf[Database::IDX_CONF_DT];
	
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
	_pi0=proj_param.pi0();
}


void Conn::update_cpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	float old_prn = ptr_conf1[Database::IDX_CONF_OLD_PRN];
	if(old_prn!=prn){
			// row update : update i (ZEPi)
		float *ptr_pi = _pi->mutable_cpu_data();
		float *ptr_ei = _ei->mutable_cpu_data();
		float *ptr_zi = _zi->mutable_cpu_data();
		int *ptr_ti = _ti->mutable_cpu_data();

		for(int i=0; i<_h; i++){
			full_update_i_kernel_cpu(
				i,
				ptr_pi,
				ptr_ei,
				ptr_zi,
				ptr_ti,
				simstep,
				_taupdt*old_prn,
				_tauedt,
				_tauzidt
			);
		}

		// row update: update ij (ZEPij, wij, epsc)
		float *ptr_pj = _pj->mutable_cpu_data();
		float *ptr_pij = _pij->mutable_cpu_data();
		float *ptr_eij = _eij->mutable_cpu_data();
		float *ptr_zi2 = _zi2->mutable_cpu_data();
		float *ptr_zj2 = _zj2->mutable_cpu_data();
		int *ptr_tij = _tij->mutable_cpu_data();
		float *ptr_wij = _wij->mutable_cpu_data();
	
		for(int i=0; i<_h*_w; i++){
			full_update_ij_kernel_cpu(
				i,
				_w,
				ptr_pi,
				ptr_pj,
				ptr_pij,
				ptr_eij,
				ptr_zi2,
				ptr_zj2,
				ptr_tij,
				ptr_wij,
				simstep,
				_taupdt*old_prn,
				_tauedt,
				_tauzidt,
				_tauzjdt,
				_wgain,
				_eps,
				_eps2
			);
		}
	}
	
	
	// get active in spike
	CONST_HOST_VECTOR(int, *v_spike) = _spike->cpu_vector();
	CONST_HOST_VECTOR(int, *v_ii) = _ii->cpu_vector();
	CONST_HOST_VECTOR(int, *v_di) = _di->cpu_vector();
	HOST_VECTOR(int, *v_qi) = _qi->mutable_cpu_vector();
	HOST_VECTOR(int, *v_si) = _si->mutable_cpu_vector();
	HOST_VECTOR(int, *v_ssi) = _ssi->mutable_cpu_vector();
	
	v_ssi->clear();
//	cout << "v_ii=";
	for(int i=0; i<_h; i++){
		
		(*v_qi)[i] >>= 1;
		if((*v_qi)[i] & 0x01){
			(*v_si)[i] = 1;
			v_ssi->push_back(i);
		}else{
			(*v_si)[i] = 0;
		}
	
		int spk = (*v_spike)[(*v_ii)[i]];
		if(spk){
			(*v_qi)[i] |= (0x01 << (*v_di)[i]);
		}
//		cout << (*v_ii)[i] << ",";
	}
//	cout << endl;

	
	// get active out spike
	HOST_VECTOR(int, *v_sj) = _sj->mutable_cpu_vector();
	HOST_VECTOR(int, *v_ssj) = _ssj->mutable_cpu_vector();
	v_ssj->clear();
	for(int i=0; i<_w; i++){
		if((*v_spike)[i+_mcu_start]){
			(*v_sj)[i] = 1;
			v_ssj->push_back(i);
		}else{
			(*v_sj)[i] = 0;
		}
	}
	
	int active_row_num = v_ssi->size();
	int active_col_num = v_ssj->size();
	const int *ptr_ssi = _ssi->cpu_data();
	const int *ptr_ssj = _ssj->cpu_data();
	
	// row update : update i (ZEPi)
	float *ptr_pi = _pi->mutable_cpu_data();
	float *ptr_ei = _ei->mutable_cpu_data();
	float *ptr_zi = _zi->mutable_cpu_data();
	int *ptr_ti = _ti->mutable_cpu_data();

	for(int i=0; i<active_row_num; i++){
		update_i_kernel_cpu(
			i,
			ptr_ssi,
			ptr_pi,
			ptr_ei,
			ptr_zi,
			ptr_ti,
			simstep,
			_taupdt*prn,
			_tauedt,
			_tauzidt,
			_kfti
		);
	}

	// full update : update j (ZEPj, bj)
	float *ptr_pj = _pj->mutable_cpu_data();
	float *ptr_ej = _ej->mutable_cpu_data();
	float *ptr_zj = _zj->mutable_cpu_data();
	float *ptr_bj = _bj->mutable_cpu_data()+_proj_start;
	float *ptr_epsc = _epsc->mutable_cpu_data()+_proj_start;
	const int *ptr_sj = _sj->cpu_data();
	for(int i=0; i<_w; i++){
		update_j_kernel_cpu(
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
			_kftj,
			_bgain,
			_eps
		);
	}
	// row update: update ij (ZEPij, wij, epsc)
	float *ptr_pij = _pij->mutable_cpu_data();
	float *ptr_eij = _eij->mutable_cpu_data();
	float *ptr_zi2 = _zi2->mutable_cpu_data();
	float *ptr_zj2 = _zj2->mutable_cpu_data();
	int *ptr_tij = _tij->mutable_cpu_data();
	float *ptr_wij = _wij->mutable_cpu_data();
	
	for(int i=0; i<active_row_num*_w; i++){
		update_ij_row_kernel_cpu(
			i,
			_w,
			ptr_ssi,
			ptr_pi,
			ptr_pj,
			ptr_pij,
			ptr_eij,
			ptr_zi2,
			ptr_zj2,
			ptr_tij,
			ptr_wij,
			ptr_epsc,
			simstep,
			_taupdt*prn,
			_tauedt,
			_tauzidt,
			_tauzjdt,
			_kfti,
			_wgain,
			_eps,
			_eps2
		);
	}

	// col update: update ij (ZEPij)
	for(int i=0; i<active_col_num*_h; i++){
		update_ij_col_kernel_cpu(
			i,
			_h,
			_w,
			ptr_ssj,
			ptr_pij,
			ptr_eij,
			ptr_zi2,
			ptr_zj2,
			ptr_tij,
			simstep,
			_taupdt*prn,
			_tauedt,
			_tauzidt,
			_tauzjdt,
			_kftj
		);
	}
}


void Conn::add_row_cpu(int src_mcu, int delay){
	_h++;
	_ii->mutable_cpu_vector()->push_back(src_mcu);
	_qi->mutable_cpu_vector()->push_back(0);
	_di->mutable_cpu_vector()->push_back(delay);
	_si->mutable_cpu_vector()->push_back(0);
	_pi->mutable_cpu_vector()->push_back(_pi0);
	_ei->mutable_cpu_vector()->push_back(0);
	_zi->mutable_cpu_vector()->push_back(0);
	_ti->mutable_cpu_vector()->push_back(0);
	_pij->mutable_cpu_vector()->resize(_h*_w, _pi0/_w);
	_eij->mutable_cpu_vector()->resize(_h*_w, 0);
	_zi2->mutable_cpu_vector()->resize(_h*_w, 0);
	_zj2->mutable_cpu_vector()->resize(_h*_w, 0);
	_tij->mutable_cpu_vector()->resize(_h*_w, 0);
	_wij->mutable_cpu_vector()->resize(_h*_w, 0);

}
/*
void Conn::add_row_gpu(int src_mcu, int delay){
	_h++;
	//_ii->mutable_gpu_vector()->push_back(src_mcu);
	//_qi->mutable_gpu_vector()->push_back(0);
	//_di->mutable_gpu_vector()->push_back(delay);
	//_si->mutable_gpu_vector()->push_back(0);
	//_pi->mutable_gpu_vector()->push_back(_pi0);
	//_ei->mutable_gpu_vector()->push_back(0);
	//_zi->mutable_gpu_vector()->push_back(0);
	//_ti->mutable_gpu_vector()->push_back(0);
	_pij->mutable_gpu_vector()->resize(_h*_w, _pi0/_w);
	_eij->mutable_gpu_vector()->resize(_h*_w, 0.0);
	_zi2->mutable_gpu_vector()->resize(_h*_w, 0.0);
	_zj2->mutable_gpu_vector()->resize(_h*_w, 0.0);
	_tij->mutable_gpu_vector()->resize(_h*_w, 0);
	_wij->mutable_gpu_vector()->resize(_h*_w, 0.0);

}
*/

void full_update_i_kernel_cpu(
	int idx,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	int simstep,
	float kp,
	float ke,
	float kzi
){
	int index = idx;
	float zi = ptr_zi[index];
	int ti = ptr_ti[index];
	int pdt = simstep - ti;
	if(pdt<=0){
		ptr_ti[index]=simstep;
		return;
	}
	
	float pi = ptr_pi[index];
	float ei = ptr_ei[index];
	
	pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
		(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
		((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
		(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
	ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
		(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
	zi = zi*exp(-kzi*pdt);
	ti = simstep;
		
	ptr_pi[index] = pi;
	ptr_ei[index] = ei;
	ptr_zi[index] = zi;
	ptr_ti[index] = ti;
}

void full_update_ij_kernel_cpu(
	int idx,
	int w,
	const float *ptr_pi,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float *ptr_wij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float wgain,
	float eps,
	float eps2
){
	int row = idx/w;
	int col = idx%w;
	int index = idx;
	
	float pij = ptr_pij[index];
	int tij = ptr_tij[index];
	float zi = ptr_zi2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_tij[index]=simstep;
	}else{
		float eij = ptr_eij[index];
		float zj = ptr_zj2[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
			(ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
			(ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi = zi*exp(-kzi*pdt);
		zj = zj*exp(-kzj*pdt);
		tij = simstep;
			 	
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi;
		ptr_zj2[index] = zj;
		ptr_tij[index] = tij;
	}
	
	// update wij and epsc
	float wij;
	if(kp){
		float pj = ptr_pj[col];
		float pi = ptr_pi[row];
		wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
		ptr_wij[index] = wij;
	}
}

void update_i_kernel_cpu(
	int idx,
	const int *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kfti
){
	int index = ptr_ssi[idx];
	float zi = ptr_zi[index];
	int ti = ptr_ti[index];
	int pdt = simstep - ti;
	if(pdt<=0){
		zi += kfti;
		ptr_ti[index]=simstep;
		return;
	}
	
	float pi = ptr_pi[index];
	float ei = ptr_ei[index];
	
	
	pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
		(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
		((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
		(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
	ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
		(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
	zi = zi*exp(-kzi*pdt) + kfti;
	ti = simstep;
	ptr_pi[index] = pi;
	ptr_ei[index] = ei;
	ptr_zi[index] = zi;
	ptr_ti[index] = ti;

}

void update_j_kernel_cpu(
	int idx,
	const int *ptr_sj,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_bj,
	float *ptr_epsc,
	float kp,
	float ke,
	float kzj,
	float kzi,
	float kftj,
	float bgain,
	float eps
){
	int index = idx;	
	
	float pj = ptr_pj[index];
	float ej = ptr_ej[index];
	float zj = ptr_zj[index];
	float sj = ptr_sj[index];
	
	// update epsc: decrease with kzi
	ptr_epsc[index] *= (1-kzi);
	
	// update bj
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj[index]=bj;
	}
	
	// update ZEP
	pj += (ej - pj)*kp;
	ej += (zj - ej)*ke;
	if(sj>0){
		zj *= (1-kzj) + kftj;
	}else{
		zj *= (1-kzj);
	}
		
	ptr_pj[index] = pj;
	ptr_ej[index] = ej;
	ptr_zj[index] = zj;

}

void update_ij_row_kernel_cpu(
	int idx,
	int w,
	const int *ptr_ssi,
	const float *ptr_pi,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float *ptr_wij,
	float *ptr_epsc,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfti,
	float wgain,
	float eps,
	float eps2
){
	int row = ptr_ssi[idx/w];
	int col = idx%w;
	int index = row*w+col;
	
	float pij = ptr_pij[index];
	int tij = ptr_tij[index];
	float zi = ptr_zi2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		zi += kfti;
		ptr_zi2[index]=zi;
		ptr_tij[index]=simstep;
		
	}else{
		float eij = ptr_eij[index];
		float zj = ptr_zj2[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
			(ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
			(ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi = zi*exp(-kzi*pdt)+kfti;
		zj = zj*exp(-kzj*pdt);
		tij = simstep;
			 	
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi;
		ptr_zj2[index] = zj;
		ptr_tij[index] = tij;
	}
	
	// update wij and epsc
	float wij;
	if(kp){
		float pj = ptr_pj[col];
		float pi = ptr_pi[row];
		wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
		ptr_wij[index] = wij;
	}else{
		wij = ptr_wij[index];
	}
	
	// update epsc
	ptr_epsc[col] += wij;
}

void update_ij_col_kernel_cpu(
	int idx,
	int h,
	int w,
	const int *ptr_ssj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kftj
){
	int row = idx%h;
	int col = ptr_ssj[idx/h];
	int index = row*w+col;
	
	int tij = ptr_tij[index];
	float zj = ptr_zj2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		zj += kftj;
		ptr_zj2[index]=zj;
		ptr_tij[index]=simstep;
		return;
	}
	
	float pij = ptr_pij[index];
	float eij = ptr_eij[index];
	float zi = ptr_zi2[index];
	
	pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
		(ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
		((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
		(ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
		(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
	eij = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
		(ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
	zi = zi*exp(-kzi*pdt);
	zj = zj*exp(-kzj*pdt)+kftj;
	tij = simstep;
		 	
	ptr_pij[index] = pij;
	ptr_eij[index] = eij;
	ptr_zi2[index] = zi;
	ptr_zj2[index] = zj;
	ptr_tij[index] = tij;
		
	// update wij
	// no need to update wij, since wij only useful when row is active.
}

}
}
