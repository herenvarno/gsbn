#include "gsbn/procedures/ProcNetGroup/Proj.hpp"

namespace gsbn{
namespace proc_net_group{

void Proj::init_new(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn){
	_list_proj = list_proj;
	_list_proj->push_back(this);
	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];
	Pop* p= (*list_pop)[proj_param.dest_pop()];
	int hcu_start = p->_hcu_start;
	int hcu_num = p->_hcu_num;
	for(int i=hcu_start; i<hcu_start+hcu_num; i++){
		Hcu *h = (*list_hcu)[i];
		Group *g = (*list_group)[h->_group_id];
		
		Conn* c = new Conn();
		c->init_new(proj_param, db, list_conn, h->_mcu_num);
		
		CHECK(c->_epsc = db.sync_vector_f("epsc_"+to_string(g->_id)));
		CHECK(c->_bj = db.sync_vector_f("bj_"+to_string(g->_id)));
		c->_proj_start=(h->_isp.size())*(g->_mcu_num)+(h->_id-g->_hcu_start)*(h->_mcu_num);
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
		
		if(h->_isp.size()>g->_conn_num){
			g->_conn_num = h->_isp.size();
			g->_epsc->mutable_cpu_vector()->resize(g->_conn_num*g->_mcu_num);
			g->_bj->mutable_cpu_vector()->resize(g->_conn_num*g->_mcu_num);
		}
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

}
}
