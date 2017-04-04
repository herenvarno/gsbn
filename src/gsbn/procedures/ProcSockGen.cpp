#include "gsbn/procedures/ProcSockGen.hpp"

namespace gsbn{
namespace proc_sock_gen{

REGISTERIMPL(ProcSockGen);

void ProcSockGen::init_new(SolverParam solver_param, Database& db){
	
	_db = &db;
	
	GenParam gen_param = solver_param.gen_param();
	_eps = gen_param.eps();
	
	float dt = gen_param.dt();
	_glv.putf("dt", dt);
	_glv.putb("plasticity", true);
	_glv.puti("cycle-flag", 1);
	_glv.puti("wmask-idx", 0);
	_glv.puti("lginp-idx", 0);
	_glv.putf("prn", 0);
	
	
	CHECK(_wmask = db.create_sync_vector_f32(".wmask"));
	CHECK(_lginp = db.create_sync_vector_f32(".lginp"));
	
	int mcu_num=0;
	int hcu_num=0;
	int shcu_num=0;
	int mhcu_num=0;
	int smcu_num=0;
	int mmcu_num=0;
	NetParam net_param=solver_param.net_param();
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		_pop_spike.push_back(NULL);
		_pop_dim_hcu.push_back(pop_param.hcu_num());
		_pop_dim_mcu.push_back(pop_param.mcu_num());
		_pop_mcu_start.push_back( mcu_num);
		_pop_hcu_start.push_back(hcu_num);
		mcu_num+=(pop_param.hcu_num()*pop_param.mcu_num());
		hcu_num+=(pop_param.hcu_num());
		if(pop_param.type()==1){
			_sp.push_back(i);
			_wmask->resize(hcu_num, 0);
			shcu_num+=pop_param.hcu_num();
			smcu_num+=pop_param.hcu_num()*pop_param.mcu_num();
		}else if(pop_param.type()==2){
			_mp.push_back(i);
			_wmask->resize(hcu_num, 1);
			mhcu_num+=pop_param.hcu_num();
			mmcu_num+=pop_param.hcu_num()*pop_param.mcu_num();
		}else{
			_wmask->resize(hcu_num, 1);
		}
	}
	_lginp->resize(mcu_num);
	
	ProcParam proc_param = get_proc_param(solver_param);
	Parser par(proc_param);
	string hostname;
	if(!par.args("hostname", hostname)){
		hostname = "127.0.0.1";
	}
	_server.sin_addr.s_addr = inet_addr(hostname.c_str());
	int port;
	CHECK(par.argi("port", port));
	_server.sin_port = htons(port);
	_server.sin_family = AF_INET;
	_sv.resize(shcu_num);
	_av.resize(mhcu_num);
	_av_cnt.resize(mmcu_num);
	
	CHECK(par.args("name", _name));
	
	_prev_proc = 1;
}

void ProcSockGen::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

void ProcSockGen::update_cpu(){
	_glv.puti("cycle-flag", 1);
	
	int _current_step;
	CHECK(_glv.geti("simstep", _current_step));
	_current_step++;
	_glv.puti("simstep", _current_step);
	
//	if(_current_step>1000){
//		_glv.puti("cycle-flag", -1);
//	}
	
	//FIXME
	// consider _spike_buffer_size
	
	int idx=0;
	for(int i=0; i<_mp.size(); i++){
		int pop = _mp[i];
		SyncVector<int8_t>* spk=_pop_spike[i];
		if(!spk){
			spk = _db->sync_vector_i8("spike_"+to_string(pop));
			_pop_spike[i]=spk;
		}
		const int8_t *ptr_spk = spk->cpu_data();
		int size = spk->size();
		for(int j=0; j<size; j++){
			_av_cnt[idx++]+=int(ptr_spk[j]>0);
		}
	}
	
	if(_current_step%100==0){
		if(_prev_proc==0){
			/*
			 * in next short period of time, the weight will be updated according to prn
			 */
		
		
			int idx_av=0;
			int idx_cnt=0;
			for(int i=0; i<_mp.size(); i++){
				int pop = _mp[i];
				for(int j=0; j<_pop_dim_hcu[pop]; j++){
					_av[idx_av]=-1;
					int max=0;
					for(int k=0; k<_pop_dim_mcu[pop]; k++){
						if(_av_cnt[idx_cnt]>max){
							_av[idx_av]=k;
							max=_av_cnt[idx_cnt];
						}
						_av_cnt[idx_cnt]=0;
						idx_cnt++;
					}
					if(max==0){
						return;
					}
					idx_av++;
				}
			}
			
			
			
			int sock = socket(AF_INET, SOCK_STREAM, 0);
			CHECK_GE(sock, 0);
			CHECK_GE(connect(sock, (struct sockaddr *)&_server, sizeof(_server)), 0);
			string str_snd = _name;
			for(int i=0; i<_av.size(); i++){
				str_snd = str_snd + "," + to_string(_av[i]);
			}
			LOG(INFO) << "SEND AV: " << str_snd;
			CHECK_GE(send(sock, str_snd.c_str(), str_snd.length(), 0), 0);
			char buffer[1024]={0};
			CHECK_GE(recv(sock, buffer, sizeof(buffer), 0), 0);
			LOG(INFO) << "RECV PRN: " << buffer;
			close(sock);
		
			if(strlen(buffer)==0){
				_glv.puti("cycle-flag", -1);
				return;
			}
		
			string str_rcv=buffer;
			float prn = stof(str_rcv);
			float old_prn;
			CHECK(_glv.getf("prn", old_prn));
			_glv.putf("old-prn", old_prn);
			_glv.putf("prn", prn);
			
			float *ptr_lginp = _lginp->mutable_cpu_data();
			float *ptr_wmask = _wmask->mutable_cpu_data();
			idx_av=0;
			for(int i=0; i<_mp.size(); i++){
				int pop = _mp[i];
				for(int j=0; j<_pop_dim_hcu[pop]; j++){
					int offset_hcu = _pop_hcu_start[pop];
					int offset = _pop_mcu_start[pop]+j*_pop_dim_mcu[pop];
					for(int k=0; k<_pop_dim_mcu[pop]; k++){
						if(_av[idx_av]==k){
							ptr_lginp[offset+k] = log(1+_eps);
						}else{
							ptr_lginp[offset+k] = log(0+_eps);
						}
					}
					ptr_wmask[offset_hcu+j]=0;
					idx_av++;
				}
			}
			
			_prev_proc = 1;
			LOG(INFO) << "NOW PRN=" << prn;
			
		}else{
		
			/*
			 * in next short period of time, weight will be fixed, and produce new action
			 */
		
		
			int sock = socket(AF_INET, SOCK_STREAM, 0);
			CHECK_GE(sock, 0);
			CHECK_GE(connect(sock, (struct sockaddr *)&_server, sizeof(_server)), 0);
			string str_snd = _name;
			LOG(INFO) << "SEND REQ: " << str_snd;
			CHECK_GE(send(sock, str_snd.c_str(), str_snd.length(), 0), 0);
			char buffer[1024]={0};
			CHECK_GE(recv(sock, buffer, sizeof(buffer), 0), 0);
			LOG(INFO) << "RECV SV: " << buffer;
			close(sock);
		
			string str_rcv=buffer;
			vector<string> sensor_list;
			split(str_rcv , ',', back_inserter(sensor_list));
		
			float prn = stof(sensor_list[0]);
			float old_prn;
			CHECK(_glv.getf("prn", old_prn));
			_glv.putf("old-prn", old_prn);
			_glv.putf("prn", 0);
		
			for(int i=0; i<sensor_list.size(); i++){
				_sv[i] = stoi(sensor_list[i]);
			}
		
			float *ptr_lginp = _lginp->mutable_cpu_data();
			float *ptr_wmask = _wmask->mutable_cpu_data();
			int idx_sv=0;
			for(int i=0; i<_sp.size(); i++){
				int pop = _sp[i];
				for(int j=0; j<_pop_dim_hcu[pop]; j++){
					int offset = _pop_mcu_start[pop]+j*_pop_dim_mcu[pop];
					for(int k=0; k<_pop_dim_mcu[pop]; k++){
						if(_sv[idx_sv]==k){
							ptr_lginp[offset+k] = log(1+_eps);
						}else{
							ptr_lginp[offset+k] = log(0+_eps);
						}
					}
					idx_sv++;
				}
			}
			for(int i=0; i<_mp.size(); i++){
				int pop = _mp[i];
				for(int j=0; j<_pop_dim_hcu[pop]; j++){
					int offset_hcu = _pop_hcu_start[pop];
					int offset = _pop_mcu_start[pop]+j*_pop_dim_mcu[pop];
					for(int k=0; k<_pop_dim_mcu[pop]; k++){
						ptr_lginp[offset+k] = log(0+_eps);
					}
					ptr_wmask[offset_hcu+j]=1;
				}
			}
		
			_prev_proc = 0;
			LOG(INFO) << "NOW PRN=" << 0;
		}
	}
}

#ifndef CPU_ONLY
void ProcSockGen::update_gpu(){
	update_cpu();
}
#endif
}
}
