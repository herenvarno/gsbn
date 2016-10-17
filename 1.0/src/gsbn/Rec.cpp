#include "gsbn/Rec.hpp"

namespace gsbn{

Rec::Rec(): _directory(), _period(1), _db(NULL), _conf(NULL){
}

void Rec::init(Database& db){
	CHECK(_conf = db.table("conf"));
	_db = &db;
}


void Rec::set_directory(string dir){
	CHECK(!dir.empty());
	
	struct stat info;
	/* Check directory exists */
	if( stat( dir.c_str(), &info ) == 0 && (info.st_mode & S_IFDIR)){
		_directory = dir;
	} else {
		LOG(WARNING) << "Directory does not exist! Create one!";
		string cmd="mkdir -p "+dir;
		if(system(cmd.c_str())!=0){
			LOG(FATAL) << "Cannot create directory for state records! Aboart!";
		}
		_directory = dir;
	}
}

void Rec::set_period(int period){
	_period = period;
}

void Rec::record(bool force){
	CHECK(!_directory.empty());
	
	if((!force) && (_period<=0)){
		return;
	}
	
	int simstep = static_cast<const int *>(_conf->cpu_data(0))[Database::IDX_CONF_TIMESTAMP];
	float dt = static_cast<const float *>(_conf->cpu_data(0))[Database::IDX_CONF_DT];
	CHECK_GE(simstep, 0);
	CHECK_GE(dt, 0);
	
	if((!force) && simstep%_period!=0){
		return;
	}
	
	SolverState st = _db->state_to_proto();
	st.set_timestamp(simstep*dt);
/*	
	string filename = _directory+"/SolverState_"+to_string(simstep*dt)+".bin";
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	if (!st.SerializeToOstream(&output)) {
		LOG(FATAL) << "Failed to write states.";
	}
	*/
	string filename = _directory+"/SolverState.txt";
	fstream output1(filename, ios::out | ios::app);
	output1 << simstep*dt;
	SyncVector<int>* spike=_db->sync_vector_i("spike");
	int size=spike->cpu_vector()->size();
	for(int i=0; i<size; i++){
		if((*(spike->cpu_vector()))[i]!=0){
			output1 << ","<<i;
		}
	}
	output1<<endl;
	
	if(simstep*dt > 5.4 and simstep*dt < 5.5){
		string filename2 = _directory+"/SolverState_"+to_string(simstep*dt)+"_wij.txt";
		fstream output2(filename2, ios::out | ios::trunc);
		for(int i=0; i<10; i++){
			CONST_HOST_VECTOR(int, *v_ii);
			CHECK(v_ii=_db->sync_vector_i("ii_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_pij);
			CHECK(v_pij=_db->sync_vector_f("pij_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_eij);
			CHECK(v_eij=_db->sync_vector_f("eij_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_zi2);
			CHECK(v_zi2=_db->sync_vector_f("zi2_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_zj2);
			CHECK(v_zj2=_db->sync_vector_f("zj2_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(int, *v_tij);
			CHECK(v_tij=_db->sync_vector_i("tij_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_wij);
			CHECK(v_wij=_db->sync_vector_f("wij_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_epsc);
			CHECK(v_epsc=_db->sync_vector_f("epsc_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_bj);
			CHECK(v_bj=_db->sync_vector_f("bj_"+to_string(i))->cpu_vector());
			for(int j=0; j<v_ii->size(); j++){
				for(int k=0; k<10; k++){
					output2 << (*v_ii)[j] <<"," << i*10+k << ","<<(*v_pij)[j*10+k]<<","<<(*v_eij)[j*10+k]<<","<<(*v_zi2)[j*10+k]<<","<<(*v_zj2)[j*10+k]<<","<<(*v_tij)[j*10+k]<<","<<(*v_wij)[j*10+k]<<"," <<(*v_epsc)[k]<<"," <<(*v_bj)[k]<<endl;
				}
			}
		}
	}
	if(simstep*dt > 5.4 and simstep*dt < 5.5){
		string filename3 = _directory+"/SolverState_"+to_string(simstep*dt)+"_sup.txt";
		fstream output3(filename3, ios::out | ios::trunc);
		for(int i=0; i<10; i++){
			CONST_HOST_VECTOR(float, *v_dsup);
			CHECK(v_dsup=_db->sync_vector_f("dsup_"+to_string(i))->cpu_vector());
			CONST_HOST_VECTOR(float, *v_act);
			CHECK(v_act=_db->sync_vector_f("act_"+to_string(i))->cpu_vector());
			for(int j=0; j<10; j++){
				output3 << i <<"," << i*10+j << ","<<(*v_dsup)[j]<<"," <<(*v_act)[j]<<endl;
			}
		}
	}
}

}
