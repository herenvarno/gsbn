#include "gsbn/procedures/ProcUpdLazyAda/Proj.hpp"

namespace gsbn{
namespace proc_upd_lazy_ada{

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
	p->_epsc->mutable_cpu_vector()->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	p->_bj->mutable_cpu_vector()->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	float dt;
	CHECK(_glv.getf("dt", dt));

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

	CHECK(_ii = db.create_sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.create_sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.create_sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_siq = db.create_sync_vector_f32(".siq_"+to_string(_id)));
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
	CHECK(_zi2 = db.create_sync_vector_f32("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.create_sync_vector_f32("zj2_"+to_string(_id)));
	CHECK(_fp = db.create_sync_vector_f32("fp_"+to_string(_id)));
	CHECK(_tij = db.create_sync_vector_i32("tij_"+to_string(_id)));
	CHECK(_wij = db.create_sync_vector_f32("wij_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	_ii->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, -1);
	_qi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_di->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, 1.0/_dim_conn);
	_ei->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_zi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_ti->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_siq->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, 1.0/_dim_mcu);
	_ej->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_zj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_pij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu, 1.0/_dim_conn/_dim_mcu);
	_eij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zi2->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zj2->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_fp->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu, 1.0);
	_tij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_wij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	
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
	CHECK_LE(p->_epsc->cpu_vector()->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	CHECK_LE(p->_bj->mutable_cpu_vector()->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	float dt;
	CHECK(_glv.getf("dt", dt));

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

	CHECK(_ii = db.sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_siq = db.create_sync_vector_f32(".siq_"+to_string(_id)));
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
	CHECK(_zi2 = db.sync_vector_f32("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.sync_vector_f32("zj2_"+to_string(_id)));
	CHECK(_tij = db.sync_vector_i32("tij_"+to_string(_id)));
	CHECK(_wij = db.sync_vector_f32("wij_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	CHECK_EQ(_ii->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_qi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_di->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_pi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ei->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_zi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ti->cpu_vector()->size(), _dim_hcu * _dim_conn);
	_siq->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	CHECK_EQ(_pj->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_ej->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_zj->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_pij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_eij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_zi2->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_zj2->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_tij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_wij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	
}

void update_full_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	float *ptr_fp,
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
	if(j==0){
		float pi = ptr_pi[i];
		float zi = ptr_zi[i];
		int ti = ptr_ti[i];
		int pdt = simstep - ti;
		if(pdt>0){
			float ei = ptr_ei[i];
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt);
			ti = simstep;
			ptr_pi[i] = pi;
			ptr_ei[i] = ei;
			ptr_zi[i] = zi;
			ptr_ti[i] = ti;
		}
	}
	
	int index = i*dim_mcu+j;
	
	float pij = ptr_pij[index];
	int tij = ptr_tij[index];
	float zi2 = ptr_zi2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_tij[index]=simstep;
	}else{
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zj2 = ptr_zj2[index];
		float fp = ptr_fp[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
			(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
			(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi2 = zi2*exp(-kzi*pdt);
		zj2 = zj2*exp(-kzj*pdt);
		tij = simstep;
		
		for(int s=0;s<pdt;s++){
			fp += (1-fp)*0.2;
		}
		fp += 0.2;
		
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi2;
		ptr_zj2[index] = zj2;
		ptr_tij[index] = tij;
		ptr_fp[index] = fp;
			
		// update wij and epsc
		float wij;
		if(kp){
			float pi = ptr_pi[i];
			float pj = ptr_pj[i/dim_conn*dim_mcu + j];
			//wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			if(pi<eps || pj<eps){
				wij=0;
			}else{
				wij = wgain * log(pij/(pi*pj));
			}
			ptr_wij[index] = wij;
		}
	}
}

void update_j_kernel_cpu(
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
		float bj;
			if(pj<eps){
				bj = bgain * log(eps);
			}else{
				bj = bgain * log(pj);
			}
		ptr_bj[idx] = bj;
	}


	ptr_pj[idx] = pj;
	ptr_ej[idx] = ej;
	ptr_zj[idx] = zj;
}

void update_row_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	const int *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	float *ptr_fp,
	int *ptr_tij,
	float* ptr_wij,
	float* ptr_epsc,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfp,
	float kfti,
	float wgain,
	float eps,
	float eps2
){
	int row = ptr_ssi[i];
	int col = j;
	int index = row*dim_mcu+col;
	
	if(j==0){
		float pi = ptr_pi[row];
		float ei = ptr_ei[row];
		float zi = ptr_zi[row];
		int ti = ptr_ti[row];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_zi[row] += kfti;
			ptr_ti[row] = simstep;
		}else{
			float pi = ptr_pi[row];
			float ei = ptr_ei[row];
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt) + kfti;
			ti = simstep;
			ptr_pi[row] = pi;
			ptr_ei[row] = ei;
			ptr_zi[row] = zi;
			ptr_ti[row] = ti;
		}
	}
	
	float pij = ptr_pij[index];
	int tij = ptr_tij[index];
	float zi2 = ptr_zi2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_zi2[index] += kfti;
		ptr_tij[index] = simstep;
	}else{
		float eij = ptr_eij[index];
		float zj2 = ptr_zj2[index];
		float fp = ptr_fp[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
			(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
			(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi2 = zi2*exp(-kzi*pdt)+kfti;
		zj2 = zj2*exp(-kzj*pdt);
		tij = simstep;
		
		for(int s=0;s<pdt;s++){
			fp += (1-fp)*0.2;
		}
		fp += 0.2;
//		cout << fp << endl;
			
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi2;
		ptr_zj2[index] = zj2;
		ptr_tij[index] = tij;
		ptr_fp[index] = fp;
		
		float wij;
		int idx_hcu = row / dim_conn;
		int idx_mcu = idx_hcu * dim_mcu + j;
		if(kp){
			float pi = ptr_pi[row];
			float pj = ptr_pj[idx_mcu];
			//wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			if(pi<eps || pj<eps){
				wij=0;
			}else{
				wij = wgain * log(pij/(pi*pj));
			}
			ptr_wij[index] = wij;
		}else{
			wij = ptr_wij[index];
		}
		
		ptr_epsc[idx_mcu] += wij*fp*zi2;
	}
}

void update_col_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	const int *ptr_ii,
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
	int row = ptr_ssj[j]/dim_mcu*dim_conn+i;
	if(ptr_ii[row]<0){
		return;
	}
	int col = ptr_ssj[j]%dim_mcu;
	int index = row*dim_mcu+col;
	
	int tij = ptr_tij[index];
	float zj2 = ptr_zj2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		zj2 += kftj;
		ptr_zj2[index]=zj2;
		ptr_tij[index]=simstep;
	}else{
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zi2 = ptr_zi2[index];
		
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
			(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
			(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi2 = zi2*exp(-kzi*pdt);
		zj2 = zj2*exp(-kzj*pdt)+kftj;
		tij = simstep;
		
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi2;
		ptr_zj2[index] = zj2;
		ptr_tij[index] = tij;
	}
}

void Proj::update_full_cpu(){
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
		float *ptr_zi2 = _zi2->mutable_cpu_data();
		float *ptr_zj2 = _zj2->mutable_cpu_data();
		float *ptr_fp = _fp->mutable_cpu_data();
		int *ptr_tij = _tij->mutable_cpu_data();
		float *ptr_wij = _wij->mutable_cpu_data();

		for(int i=0; i<_dim_hcu * _dim_conn; i++){
			for(int j=0; j<_dim_mcu; j++){
				update_full_kernel_cpu(
					i,
					j,
					_dim_conn,
					_dim_mcu,
					ptr_pi,
					ptr_ei,
					ptr_zi,
					ptr_ti,
					ptr_pj,
					ptr_pij,
					ptr_eij,
					ptr_zi2,
					ptr_zj2,
					ptr_fp,
					ptr_tij,
					ptr_wij,
					simstep-1,
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
	}
}

void Proj::update_j_cpu(){
	float prn;
	CHECK(_glv.getf("prn", prn));
	float *ptr_pj = _pj->mutable_cpu_data();
	float *ptr_ej = _ej->mutable_cpu_data();
	float *ptr_zj = _zj->mutable_cpu_data();
	float *ptr_epsc = _epsc->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_bj = _bj->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int8_t *ptr_sj = _sj->mutable_cpu_data();

	for(int i=0; i<_dim_hcu * _dim_mcu; i++){
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
			_tauepscdt,
			_kftj,
			_bgain,
			_eps
		);
	}
}

void Proj::update_ss_cpu(){
	// get active in spike
	CONST_HOST_VECTOR(int8_t, *v_si) = _si->cpu_vector();
	CONST_HOST_VECTOR(int, *v_ii) = _ii->cpu_vector();
	CONST_HOST_VECTOR(int, *v_di) = _di->cpu_vector();
	HOST_VECTOR(int, *v_qi) = _qi->mutable_cpu_vector();
	HOST_VECTOR(int, *v_ssi) = _ssi->mutable_cpu_vector();
	
	v_ssi->clear();
	for(int i=0; i<_dim_conn * _dim_hcu; i++){
		if((*v_ii)[i]<0){
			continue;
		}
		(*v_qi)[i] >>= 1;
		if((*v_qi)[i] & 0x01){
			v_ssi->push_back(i);
		}
	
		int spk = (*v_si)[(*v_ii)[i]];
		if(spk>0){
			(*v_qi)[i] |= (0x01 << (*v_di)[i]);
		}
	}
	
	// get active out spike
	CONST_HOST_VECTOR(int8_t, *v_sj) = _sj->cpu_vector();
	HOST_VECTOR(int, *v_ssj) = _ssj->mutable_cpu_vector();
	v_ssj->clear();
	for(int i=0; i<v_sj->size(); i++){
		if((*v_sj)[i]>0){
			v_ssj->push_back(i);
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
	float *ptr_zi2 = _zi2->mutable_cpu_data();
	float *ptr_zj2 = _zj2->mutable_cpu_data();
	int *ptr_tij = _tij->mutable_cpu_data();
	float *ptr_wij = _wij->mutable_cpu_data();
	float *ptr_epsc = _epsc->mutable_cpu_data()+ _proj_in_pop * _dim_hcu * _dim_mcu;
	float *ptr_fp = _fp->mutable_cpu_data();
	
	const int *ptr_ssi = _ssi->cpu_data();
	int active_row_num = _ssi->cpu_vector()->size();
	for(int i=0; i<active_row_num; i++){
		for(int j=0; j<_dim_mcu; j++){
			update_row_kernel_cpu(
				i,
				j,
				_dim_conn,
				_dim_mcu,
				ptr_ssi,
				ptr_pi,
				ptr_ei,
				ptr_zi,
				ptr_ti,
				ptr_pj,
				ptr_pij,
				ptr_eij,
				ptr_zi2,
				ptr_zj2,
				ptr_fp,
				ptr_tij,
				ptr_wij,
				ptr_epsc,
				simstep,
				_taupdt*prn,
				_tauedt,
				_tauzidt,
				_tauzjdt,
				_tauzidt,
				_kfti,
				_wgain,
				_eps,
				_eps2
			);
		}
	}
}

void Proj::update_col_cpu(){
	int simstep;
	float prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("prn", prn));
	
	float *ptr_pij = _pij->mutable_cpu_data();
	float *ptr_eij = _eij->mutable_cpu_data();
	float *ptr_zi2 = _zi2->mutable_cpu_data();
	float *ptr_zj2 = _zj2->mutable_cpu_data();
	int *ptr_tij = _tij->mutable_cpu_data();
	
	const int *ptr_ii = _ii->cpu_data();
	const int *ptr_ssj = _ssj->cpu_data();
	int active_col_num = _ssj->cpu_vector()->size();
	for(int i=0; i<_dim_conn; i++){
		for(int j=0; j<active_col_num; j++){
			update_col_kernel_cpu(
				i,
				j,
				_dim_conn,
				_dim_mcu,
				ptr_ii,
				ptr_ssj,
				ptr_pij,
				ptr_eij,
				ptr_zi2,
				ptr_zj2,
				ptr_tij,
				simstep,
				_taupdt * prn,
				_tauedt,
				_tauzidt,
				_tauzjdt,
				_kftj
			);
		}
	}
}

}
}