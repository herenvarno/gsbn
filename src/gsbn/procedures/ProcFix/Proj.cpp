#include "gsbn/procedures/ProcFix/Proj.hpp"

namespace gsbn{
namespace proc_fix{

void Proj::init_new(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg, int norm_frac_bit, int p_frac_bit){

	CHECK(list_pop);
	CHECK(list_proj);
	CHECK(msg);

	_norm_frac_bit = norm_frac_bit;
	_p_frac_bit = p_frac_bit;

	_list_pop = list_pop;
	_list_proj = list_proj;
	_msg = msg;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	if(_dim_conn > p->_slot_num){
		_dim_conn = p->_slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;
	p->_epsc->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	p->_bj->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];

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
 
	CHECK(_ii = db.create_sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.create_sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.create_sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_pi = db.create_sync_vector_i16("pi_"+to_string(_id)));
	CHECK(_ei = db.create_sync_vector_i16("ei_"+to_string(_id)));
	CHECK(_zi = db.create_sync_vector_i16("zi_"+to_string(_id)));
	CHECK(_ti = db.create_sync_vector_i32("ti_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i32(".ssj_"+to_string(_id)));
	CHECK(_pj = db.create_sync_vector_i16("pj_"+to_string(_id)));
	CHECK(_ej = db.create_sync_vector_i16("ej_"+to_string(_id)));
	CHECK(_zj = db.create_sync_vector_i16("zj_"+to_string(_id)));
	CHECK(_pij = db.create_sync_vector_i16("pij_"+to_string(_id)));
	CHECK(_eij = db.create_sync_vector_i16("eij_"+to_string(_id)));
	CHECK(_zi2 = db.create_sync_vector_i16("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.create_sync_vector_i16("zj2_"+to_string(_id)));
	CHECK(_tij = db.create_sync_vector_i32("tij_"+to_string(_id)));
	CHECK(_wij = db.create_sync_vector_i16("wij_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	_ii->resize(_dim_hcu * _dim_conn, -1);
	_qi->resize(_dim_hcu * _dim_conn);
	_di->resize(_dim_hcu * _dim_conn);
	_pi->resize(_dim_hcu * _dim_conn, fp32_to_fix16(_pi0, _p_frac_bit));
	_ei->resize(_dim_hcu * _dim_conn);
	_zi->resize(_dim_hcu * _dim_conn);
	_ti->resize(_dim_hcu * _dim_conn);
	_pj->resize(_dim_hcu * _dim_mcu, fp32_to_fix16(1.0/_dim_mcu, _p_frac_bit));
	_ej->resize(_dim_hcu * _dim_mcu);
	_zj->resize(_dim_hcu * _dim_mcu);
	_pij->resize(_dim_hcu * _dim_conn * _dim_mcu, fp32_to_fix16(_pi0/_dim_mcu, _p_frac_bit));
	_eij->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zi2->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zj2->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_tij->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_wij->resize(_dim_hcu * _dim_conn * _dim_mcu);

	vector<int> list;
	for(int i=0; i<_ptr_dest_pop->_dim_hcu; i++){
		list.push_back(i);
	}
	for(int i=0; i<_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu; i++){
		_ptr_src_pop->_avail_hcu[i].push_back(list);
	}
	_ptr_src_pop->_avail_proj.push_back(_id);
	
	_conn_cnt.resize(_dim_hcu, 0);

}

void Proj::init_copy(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg, int norm_frac_bit, int p_frac_bit){

	CHECK(list_pop);
	CHECK(list_proj);
	CHECK(msg);

	_norm_frac_bit = norm_frac_bit;
	_p_frac_bit = p_frac_bit;

	_list_pop = list_pop;
	_list_proj = list_proj;
	_msg = msg;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	if(_dim_conn > p->_slot_num){
		_dim_conn = p->_slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;
	CHECK_LE(p->_epsc->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	CHECK_LE(p->_bj->mutable_cpu_vector()->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];

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

	CHECK(_ii = db.sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_pi = db.sync_vector_i16("pi_"+to_string(_id)));
	CHECK(_ei = db.sync_vector_i16("ei_"+to_string(_id)));
	CHECK(_zi = db.sync_vector_i16("zi_"+to_string(_id)));
	CHECK(_ti = db.sync_vector_i32("ti_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i32(".ssj_"+to_string(_id)));
	CHECK(_pj = db.sync_vector_i16("pj_"+to_string(_id)));
	CHECK(_ej = db.sync_vector_i16("ej_"+to_string(_id)));
	CHECK(_zj = db.sync_vector_i16("zj_"+to_string(_id)));
	CHECK(_pij = db.sync_vector_i16("pij_"+to_string(_id)));
	CHECK(_eij = db.sync_vector_i16("eij_"+to_string(_id)));
	CHECK(_zi2 = db.sync_vector_i16("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.sync_vector_i16("zj2_"+to_string(_id)));
	CHECK(_tij = db.sync_vector_i32("tij_"+to_string(_id)));
	CHECK(_wij = db.sync_vector_i16("wij_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	CHECK_EQ(_ii->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_qi->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_di->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_pi->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ei->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_zi->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ti->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_pj->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_ej->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_zj->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_pij->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_eij->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_zi2->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_zj2->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_tij->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_wij->size(), _dim_hcu * _dim_conn * _dim_mcu);

	const int *ptr_ii = _ii->cpu_data();
	_conn_cnt.resize(_dim_hcu);
	for(int i=0; i<_dim_hcu; i++){
		for(int j=0; j<_dim_conn; j++){
			if(ptr_ii[i*_dim_conn+j]<0){
				_conn_cnt[i]=j;
				break;
			}
			_conn_cnt[i]=_dim_conn;
		}
	}
	
	vector<int> list;
	for(int i=0; i<_ptr_dest_pop->_dim_hcu; i++){
		list.push_back(i);
	}
	for(int i=0; i<_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu; i++){
		vector<int> list_cpy=list;
		for(int x=0; x<_dim_hcu; x++){
			for(int y=0; y<_conn_cnt[x]; y++){
				int mcu = ptr_ii[x*_dim_conn+y];
				if(mcu==i){
					for(int z=0; z<list_cpy.size(); z++){
						if(list_cpy[z]==x){
							list_cpy.erase(list_cpy.begin()+z);
							break;
						}
					}
					break;
				}
			}
		}
		_ptr_src_pop->_avail_hcu[i].push_back(list_cpy);
	}
	
	_ptr_src_pop->_avail_proj.push_back(_id);

}

void update_full_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	fix16 *ptr_pi,
	fix16 *ptr_ei,
	fix16 *ptr_zi,
	int *ptr_ti,
	const fix16 *ptr_pj,
	fix16 *ptr_pij,
	fix16 *ptr_eij,
	fix16 *ptr_zi2,
	fix16 *ptr_zj2,
	int *ptr_tij,
	fix16 *ptr_wij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float wgain,
	float eps,
	float eps2,
	int norm_frac_bit,
	int p_frac_bit
){
	if(j==0){
		float pi = fix16_to_fp32(ptr_pi[i], p_frac_bit);
		float zi = fix16_to_fp32(ptr_zi[i], norm_frac_bit);
		int ti = ptr_ti[i];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_ti[i]=simstep;
		}else{
			float ei = fix16_to_fp32(ptr_ei[i], norm_frac_bit);
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt);
			ti = simstep;
		
			ptr_pi[i] = fp32_to_fix16(pi, p_frac_bit);
			ptr_ei[i] = fp32_to_fix16(ei, norm_frac_bit);
			ptr_zi[i] = fp32_to_fix16(zi, norm_frac_bit);
			ptr_ti[i] = ti;
		}
	}
	
	int index = i*dim_mcu+j;
	
	int tij = ptr_tij[index];
	float zi2 = fix16_to_fp32(ptr_zi2[index], norm_frac_bit);
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_tij[index]=simstep;
	}else{
		float pij = fix16_to_fp32(ptr_pij[index], p_frac_bit);
		float eij = fix16_to_fp32(ptr_eij[index], norm_frac_bit);
		float zj2 = fix16_to_fp32(ptr_zj2[index], norm_frac_bit);
	
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
			 	
		ptr_pij[index] = fp32_to_fix16(pij, p_frac_bit);
		ptr_eij[index] = fp32_to_fix16(eij, norm_frac_bit);
		ptr_zi2[index] = fp32_to_fix16(zi2, norm_frac_bit);
		ptr_zj2[index] = fp32_to_fix16(zj2, norm_frac_bit);
		ptr_tij[index] = tij;
			
		// update wij and epsc
		float wij;
		if(kp){
			float pi = fix16_to_fp32(ptr_pi[i], p_frac_bit);
			float pj = fix16_to_fp32(ptr_pj[i/dim_conn*dim_mcu + j], p_frac_bit);
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = fp32_to_fix16(wij, norm_frac_bit);
		}
	}
}

void update_j_kernel_cpu(
	int idx,
	const int8_t *ptr_sj,
	fix16 *ptr_pj,
	fix16 *ptr_ej,
	fix16 *ptr_zj,
	fix16 *ptr_bj,
	fix16 *ptr_epsc,
	float kp,
	float ke,
	float kzj,
	float kzi,
	float kftj,
	float bgain,
	float eps,
	int norm_frac_bit,
	int p_frac_bit
){
	float pj = fix16_to_fp32(ptr_pj[idx], p_frac_bit);
	float ej = fix16_to_fp32(ptr_ej[idx], norm_frac_bit);
	float zj = fix16_to_fp32(ptr_zj[idx], norm_frac_bit);
	int8_t sj = ptr_sj[idx];
	
/*	if(idx%10==0){
		LOG(INFO) << "epsc: " << ptr_epsc[idx];
	}*/
	float epsc = fix16_to_fp32(ptr_epsc[idx], norm_frac_bit);
	ptr_epsc[idx] = fp32_to_fix16(epsc*(1-kzi), norm_frac_bit);
	
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj[idx] = fp32_to_fix16(bj, norm_frac_bit);
	}
	
	pj += (ej - pj)*kp;
	ej += (zj - ej)*ke;
	zj *= (1-kzj);
	if(sj){
		zj += kftj;
	}

	ptr_pj[idx] = fp32_to_fix16(pj, p_frac_bit);
	ptr_ej[idx] = fp32_to_fix16(ej, norm_frac_bit);
	ptr_zj[idx] = fp32_to_fix16(zj, norm_frac_bit);
}

void update_row_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	const int *ptr_ssi,
	fix16 *ptr_pi,
	fix16 *ptr_ei,
	fix16 *ptr_zi,
	int *ptr_ti,
	const fix16 *ptr_pj,
	fix16 *ptr_pij,
	fix16 *ptr_eij,
	fix16 *ptr_zi2,
	fix16 *ptr_zj2,
	int *ptr_tij,
	fix16* ptr_wij,
	fix16* ptr_epsc,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfti,
	float wgain,
	float eps,
	float eps2,
	int norm_frac_bit,
	int p_frac_bit
){
	int row = ptr_ssi[i];
	int col = j;
	int index = row*dim_mcu+col;
	
	if(j==0){
		float pi = fix16_to_fp32(ptr_pi[row], p_frac_bit);
		float ei = fix16_to_fp32(ptr_ei[row], norm_frac_bit);
		float zi = fix16_to_fp32(ptr_zi[row], norm_frac_bit);
		int ti = ptr_ti[row];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_zi[row] = fp32_to_fix16(zi+kfti, norm_frac_bit);
			ptr_ti[row] = simstep;
		}else{
			float pi = fix16_to_fp32(ptr_pi[row], p_frac_bit);
			float ei = fix16_to_fp32(ptr_ei[row], norm_frac_bit);
		
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt) + kfti;
			ti = simstep;
			ptr_pi[row] = fp32_to_fix16(pi, p_frac_bit);
			ptr_ei[row] = fp32_to_fix16(ei, norm_frac_bit);
			ptr_zi[row] = fp32_to_fix16(zi, norm_frac_bit);
			ptr_ti[row] = ti;
		}
	}
	
	int tij = ptr_tij[index];
	float zi2 = fix16_to_fp32(ptr_zi2[index], norm_frac_bit);
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_zi2[index] = fp32_to_fix16(zi2+kfti, norm_frac_bit);
		ptr_tij[index] = simstep;
	}else{
		float pij = fix16_to_fp32(ptr_pij[index], p_frac_bit);
		float eij = fix16_to_fp32(ptr_eij[index], norm_frac_bit);
		float zj2 = fix16_to_fp32(ptr_zj2[index], norm_frac_bit);
	
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
			 	
		ptr_pij[index] = fp32_to_fix16(pij, p_frac_bit);
		ptr_eij[index] = fp32_to_fix16(eij, norm_frac_bit);
		ptr_zi2[index] = fp32_to_fix16(zi2, norm_frac_bit);
		ptr_zj2[index] = fp32_to_fix16(zj2, norm_frac_bit);
		ptr_tij[index] = tij;
		
		float wij;
		int idx_hcu = row / dim_conn;
		int idx_mcu = idx_hcu * dim_mcu + j;
		if(kp){
			float pi = fix16_to_fp32(ptr_pi[row], p_frac_bit);
			float pj = fix16_to_fp32(ptr_pj[idx_mcu], p_frac_bit);
			
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = fp32_to_fix16(wij, norm_frac_bit);
		}else{
			wij = fix16_to_fp32(ptr_wij[index], norm_frac_bit);
		}
		float epsc = fix16_to_fp32(ptr_epsc[idx_mcu], norm_frac_bit);
		ptr_epsc[idx_mcu] = fp32_to_fix16(epsc+wij, norm_frac_bit);
	}
}

void update_col_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	const int *ptr_ii,
	const int *ptr_ssj,
	fix16 *ptr_pij,
	fix16 *ptr_eij,
	fix16 *ptr_zi2,
	fix16 *ptr_zj2,
	int *ptr_tij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kftj,
	int norm_frac_bit,
	int p_frac_bit
){
	int row = ptr_ssj[j]/dim_mcu*dim_conn+i;
	if(ptr_ii[row]<0){
		return;
	}
	int col = ptr_ssj[j]%dim_mcu;
	int index = row*dim_mcu+col;
	
	int tij = ptr_tij[index];
	float zj2 = fix16_to_fp32(ptr_zj2[index], norm_frac_bit);
	int pdt = simstep - tij;
	if(pdt<=0){
		zj2 += kftj;
		ptr_zj2[index]=fp32_to_fix16(zj2, norm_frac_bit);
		ptr_tij[index]=simstep;
	}else{
		float pij = fix16_to_fp32(ptr_pij[index], p_frac_bit);
		float eij = fix16_to_fp32(ptr_eij[index], norm_frac_bit);
		float zi2 = fix16_to_fp32(ptr_zi2[index], norm_frac_bit);
	
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
			 	
		ptr_pij[index] = fp32_to_fix16(pij, p_frac_bit);
		ptr_eij[index] = fp32_to_fix16(eij, norm_frac_bit);
		ptr_zi2[index] = fp32_to_fix16(zi2, norm_frac_bit);
		ptr_zj2[index] = fp32_to_fix16(zj2, norm_frac_bit);
		ptr_tij[index] = tij;
	}
}





void Proj::update_full_cpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	float old_prn = ptr_conf1[Database::IDX_CONF_OLD_PRN];
	if(old_prn!=prn){
		fix16 *ptr_pi = _pi->mutable_cpu_data();
		fix16 *ptr_ei = _ei->mutable_cpu_data();
		fix16 *ptr_zi = _zi->mutable_cpu_data();
		int *ptr_ti = _ti->mutable_cpu_data();
		const fix16 *ptr_pj = _pj->cpu_data();
		fix16 *ptr_pij = _pij->mutable_cpu_data();
		fix16 *ptr_eij = _eij->mutable_cpu_data();
		fix16 *ptr_zi2 = _zi2->mutable_cpu_data();
		fix16 *ptr_zj2 = _zj2->mutable_cpu_data();
		int *ptr_tij = _tij->mutable_cpu_data();
		fix16 *ptr_wij = _wij->mutable_cpu_data();

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
					ptr_tij,
					ptr_wij,
					simstep,
					_taupdt*old_prn,
					_tauedt,
					_tauzidt,
					_tauzjdt,
					_wgain,
					_eps,
					_eps2,
					_norm_frac_bit,
					_p_frac_bit
				);
			}
		}
	}
}

void Proj::update_j_cpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];
	fix16 *ptr_pj = _pj->mutable_cpu_data();
	fix16 *ptr_ej = _ej->mutable_cpu_data();
	fix16 *ptr_zj = _zj->mutable_cpu_data();
	fix16 *ptr_epsc = _epsc->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	fix16 *ptr_bj = _bj->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
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
			_kftj,
			_bgain,
			_eps,
			_norm_frac_bit,
			_p_frac_bit
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
	
		int8_t spk = (*v_si)[(*v_ii)[i]];
		if(spk){
			(*v_qi)[i] |= (0x01 << (*v_di)[i]);
		}
	}
	
	// get active out spike
	CONST_HOST_VECTOR(int8_t, *v_sj) = _sj->cpu_vector();
	HOST_VECTOR(int, *v_ssj) = _ssj->mutable_cpu_vector();
	v_ssj->clear();
	for(int i=0; i<_dim_hcu * _dim_mcu; i++){
		if((*v_sj)[i]>0){
			v_ssj->push_back(i);
		}
	}
}

void Proj::update_row_cpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	fix16 *ptr_pi = _pi->mutable_cpu_data();
	fix16 *ptr_ei = _ei->mutable_cpu_data();
	fix16 *ptr_zi = _zi->mutable_cpu_data();
	int *ptr_ti = _ti->mutable_cpu_data();
	const fix16 *ptr_pj = _pj->cpu_data();
	fix16 *ptr_pij = _pij->mutable_cpu_data();
	fix16 *ptr_eij = _eij->mutable_cpu_data();
	fix16 *ptr_zi2 = _zi2->mutable_cpu_data();
	fix16 *ptr_zj2 = _zj2->mutable_cpu_data();
	int *ptr_tij = _tij->mutable_cpu_data();
	fix16 *ptr_wij = _wij->mutable_cpu_data();
	fix16 *ptr_epsc = _epsc->mutable_cpu_data()+ _proj_in_pop * _dim_hcu * _dim_mcu;
	
	const int *ptr_ssi = _ssi->cpu_data();
	int active_row_num = _ssi->size();

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
				_eps2,
			_norm_frac_bit,
			_p_frac_bit
			);
		}
	}
}

void Proj::update_col_cpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	fix16 *ptr_pij = _pij->mutable_cpu_data();
	fix16 *ptr_eij = _eij->mutable_cpu_data();
	fix16 *ptr_zi2 = _zi2->mutable_cpu_data();
	fix16 *ptr_zj2 = _zj2->mutable_cpu_data();
	int *ptr_tij = _tij->mutable_cpu_data();
	
	const int *ptr_ii = _ii->cpu_data();
	const int *ptr_ssj = _ssj->cpu_data();
	int active_col_num = _ssj->size();
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
				_kftj,
			_norm_frac_bit,
			_p_frac_bit
			);
		}
	}
}

void Proj::receive(){
	// RECEIVE
	const int *ptr_conf = static_cast<const int *>(_conf->cpu_data());
	int plasticity = ptr_conf[Database::IDX_CONF_PLASTICITY];
	if(!plasticity){
		return;
	}
	
	vector<msg_t> list_msg = _msg->receive(_id);
	for(vector<msg_t>::iterator it = list_msg.begin(); it!=list_msg.end(); it++){
		if(it->dest_hcu < 0 ||
			it->dest_hcu > _ptr_dest_pop->_dim_hcu ||
			it->src_mcu < 0 ||
			it->src_mcu > _ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu){
			continue;
		}
		switch(it->type){
		case 1:
			if((*(_ptr_dest_pop->_slot->mutable_cpu_vector()))[it->dest_hcu]>0){
				(*(_ptr_dest_pop->_slot->mutable_cpu_vector()))[it->dest_hcu]--;
				_msg->send(_id, it->src_mcu, it->dest_hcu, 2);
				add_row(it->src_mcu, it->dest_hcu, it->delay);
			}else{
				_msg->send(_id, it->src_mcu, it->dest_hcu, 3);
			}
			break;
		case 2:
			break;
		case 3:
			(_ptr_src_pop->_fanout->mutable_cpu_data())[it->src_mcu]++;
			break;
		default:
			break;
		}
	}
}


void Proj::add_row(int src_mcu, int dest_hcu, int delay){
	
	if(_conn_cnt[dest_hcu]<_dim_conn){
		int idx = _conn_cnt[dest_hcu];
		_ii->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=src_mcu;
		_di->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=delay;
		_conn_cnt[dest_hcu]++;
	}
}

}
}
