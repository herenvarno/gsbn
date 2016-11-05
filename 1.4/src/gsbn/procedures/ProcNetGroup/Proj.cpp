#include "gsbn/procedures/ProcNetGroup/Proj.hpp"

namespace gsbn{
namespace proc_net_group{

void Proj::init_new(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn){
	_list_proj = list_proj;
	_list_proj->push_back(this);
	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= (*list_pop)[proj_param.dest_pop()];
	_offset_in_pop = p->_proj_num * p->_mcu_num;
	_offset_in_spk = p->_mcu_start;
	_mcu_num = p->_mcu_num;
	_pj = p->_pj;
	_ej = p->_ej;
	_zj = p->_zj;
	CHECK(_sj = db.sync_vector_i("spike"));
	_epsc = p->_epsc;
	_bj = p->_bj;

	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];

	_taupdt = dt/proj_param.taup();
	_tauedt = dt/proj_param.taue();
	_tauzidt = dt/proj_param.tauzi();
	_tauzjdt = dt/proj_param.tauzj();
	_eps = dt/proj_param.taup();
	_kftj = 1/(proj_param.maxfq() * proj_param.tauzj());
	_bgain = proj_param.bgain();
	
	p->_proj_num++;
	int hcu_start = p->_hcu_start;
	int hcu_num = p->_hcu_num;
	for(int i=hcu_start; i<hcu_start+hcu_num; i++){
		Hcu *h = (*list_hcu)[i];
		Group *g = (*list_group)[h->_group_id];
		
		Conn* c = new Conn();
		c->_pj = p->_pj;
		c->_ej = p->_ej;
		c->_zj = p->_zj;
		c->_epsc = p->_epsc;
		c->_bj = p->_bj;
		c->init_new(proj_param, db, list_conn, h->_mcu_num);
		c->_proj_start=(h->_isp.size())*(p->_mcu_num)+(h->_mcu_start-p->_mcu_start);
		LOG(INFO) << "proj_start of conn "<< c->_id << " : " << c->_proj_start;
		c->_mcu_start=h->_mcu_start;
		
		h->_isp.push_back(c);
		h->_isp_mcu_start.push_back(_ptr_src_pop->_mcu_start);
		h->_isp_mcu_num.push_back(_ptr_src_pop->_mcu_num);
		vector<int> list;
		for(int k=_ptr_src_pop->_hcu_start; k<_ptr_src_pop->_hcu_start+_ptr_src_pop->_hcu_num; k++){
			list.push_back(k);
		}
		for(int j=0; j<h->_mcu_num; j++){
			h->_avail_hcu[j].insert(h->_avail_hcu[j].end(), list.begin(), list.end());
		}
		
		if(h->_isp.size() > g->_conn_num){
			g->_conn_num = h->_isp.size();
			p->_pj->mutable_cpu_vector()->resize(g->_conn_num*g->_mcu_num_in_pop);
			p->_ej->mutable_cpu_vector()->resize(g->_conn_num*g->_mcu_num_in_pop);
			p->_zj->mutable_cpu_vector()->resize(g->_conn_num*g->_mcu_num_in_pop);
			p->_epsc->mutable_cpu_vector()->resize(g->_conn_num*g->_mcu_num_in_pop);
			p->_bj->mutable_cpu_vector()->resize(g->_conn_num*g->_mcu_num_in_pop);
		}
		c->init_pj();
	}
}

void Proj::init_copy(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop,  vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn){
/*	_list_proj = list_proj;
	_list_proj->push_back(this);
	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];
	
	Pop* p= (*list_pop)[proj_param.dest_pop()];
	int hcu_start = p->_hcu_start;
	int hcu_num = p->_hcu_num;
	for(int i=hcu_start; i<hcu_num+hcu_start; i++){
		Hcu *h = (*list_hcu)[i];
		
		Conn* c = new Conn();
		c->init_copy(proj_param, db, list_conn, h->_mcu_num);
		
		CHECK(c->_epsc = db.sync_vector_f("epsc_"+to_string(h->_id)));
		CHECK(c->_bj = db.sync_vector_f("bj_"+to_string(h->_id)));
		c->_proj_start=(h->_isp.size())*(h->_mcu_num);
		c->_mcu_start=h->_mcu_start;
		
		h->_isp.push_back(c);
		h->_isp_mcu_start.push_back(_ptr_src_pop->_mcu_start);
		h->_isp_mcu_num.push_back(_ptr_src_pop->_mcu_num);
		vector<int> list;
		for(int k=_ptr_src_pop->_hcu_start; k<_ptr_src_pop->_hcu_start+_ptr_src_pop->_hcu_num; k++){
			bool flag=false;
			for(int l=0; l<c->_h; l++){
				if((*(c->_ii->cpu_vector()))[i]==k)
					flag=true;
					break;
			}
			if(flag){
				continue;
			}
			list.push_back(k);
		}
		for(int j=0; j<h->_mcu_num; j++){
			h->_avail_hcu[j].insert(h->_avail_hcu[j].end(), list.begin(), list.end());
		}
	}*/
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
	float pj = ptr_pj[idx];
	float ej = ptr_pj[idx];
	float zj = ptr_pj[idx];
	int sj = ptr_pj[idx];
	ptr_epsc[idx] *= (1-kzi);
	
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj[idx] = bj;
	}
	
	pj += (ej - pj)*kp;
	ej += (zj - ej)*ke;
	zj *= (1-kzj);
	if(sj>0){
		zj += kftj;
	}

	ptr_pj[idx] = pj;
	ptr_ej[idx] = ej;
	ptr_zj[idx] = zj;
}

void Proj::update_cpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];

	float *ptr_pj = _pj->mutable_cpu_data()+_offset_in_pop;
	float *ptr_ej = _ej->mutable_cpu_data()+_offset_in_pop;
	float *ptr_zj = _zj->mutable_cpu_data()+_offset_in_pop;
	float *ptr_epsc = _epsc->mutable_cpu_data()+_offset_in_pop;
	float *ptr_bj = _bj->mutable_cpu_data()+_offset_in_pop;

	const int *ptr_sj = _sj->mutable_cpu_data()+_offset_in_spk;

	LOG(INFO) << "tauzidt = " <<_tauzidt;
	for(int i=0; i<_mcu_num; i++){
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
}

}
}
