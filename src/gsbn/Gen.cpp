#include "gsbn/Gen.hpp"

namespace gsbn{

Gen::Gen() : _current_time(0.0), _current_mode(Gen::NOP), _max_time(-1.0), _dt(0.001), _cursor(0){
}

void Gen::init(Database& db){
	CHECK(_mode = db.table("mode"));
	CHECK(_stim = db.table("stim"));
	CHECK(_conf = db.table("conf"));
	_max_time = static_cast<const float *>(_mode->cpu_data(_mode->height()-1))[Database::IDX_MODE_END_TIME];
	_current_time = static_cast<const float *>(_conf->cpu_data())[Database::IDX_CONF_TIMESTAMP];
	_dt = static_cast<const float *>(_conf->cpu_data())[Database::IDX_CONF_DT];
}
	
void Gen::update(){
	_current_time += _dt;
	static_cast<float *>(_conf->mutable_cpu_data())[Database::IDX_CONF_TIMESTAMP]=_current_time;
	if(_current_time-_max_time>(_dt/10)){ // floating point number compare
		_current_mode = END;
		return;
	}
/*	
	int r=_mode->rows();
	int l=0, h=r-1;
	int m=(l+h)/2;
	float mode=0;
	float gain_mask=1;
	int plasticity=1;
	int stim=-1;
	while(1){
		
		float bt0, et0;
		const float *ptr = static_cast<const float*>(_mode->cpu_data(m));
		bt0 = ptr[Database::IDX_MODE_BEGIN_TIME];
		et0 = ptr[Database::IDX_MODE_END_TIME];
		LOG(INFO) << l << "," << m << "," << h << ":" << _current_time << "," <<bt0 << "," << et0;
		if(_current_time >= bt0 && _current_time <= et0){
			mode = ptr[Database::IDX_MODE_PRN];
			gain_mask = ptr[Database::IDX_MODE_GAIN_MASK];
			plasticity = ((int*)(ptr))[Database::IDX_MODE_PLASTICITY];
			stim = ((int*)(ptr))[Database::IDX_MODE_STIM];
			*static_cast<float *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_PRN))=mode;
			*static_cast<float *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_GAIN_MASK))=gain_mask;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_PLASTICITY))=plasticity;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_STIM))=stim;
			_current_mode = RUN;
			break;
		}else if(_current_time < bt0){
			h = m;
			m = (h+l)/2;
			if(h == l){
				_current_mode = NOP;
				break;
			}
		}else{
			l = m;
			m = (h+l)/2;
			if(h == l){
				_current_mode = NOP;
				break;
			}
		}
	}
	*/
	
	int r=_mode->rows();
	for(; _cursor<r; _cursor++){
		float bt0, et0;
		const float *ptr = static_cast<const float*>(_mode->cpu_data(_cursor));
		bt0 = ptr[Database::IDX_MODE_BEGIN_TIME];
		et0 = ptr[Database::IDX_MODE_END_TIME];
		if(_current_time > bt0 && _current_time <= et0){
			float prn=0;
			float gain_mask=1;
			int plasticity=1;
			int stim=0;
			prn = ptr[Database::IDX_MODE_PRN];
			gain_mask = ptr[Database::IDX_MODE_GAIN_MASK];
			plasticity = ((int*)(ptr))[Database::IDX_MODE_PLASTICITY];
			stim = ((int*)(ptr))[Database::IDX_MODE_STIM];
			*static_cast<float *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_PRN))=prn;
			*static_cast<float *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_GAIN_MASK))=gain_mask;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_PLASTICITY))=plasticity;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_STIM))=stim;
			_current_mode = RUN;
			break;
		}else if(_current_time < bt0){
			_current_mode = NOP;
			break;
		}
	}
	
	
}

void Gen::set_current_time(float timestamp){
	CHECK_GE(timestamp, 0);
	_current_time = timestamp;
	static_cast<float *>(_conf->mutable_cpu_data())[Database::IDX_CONF_TIMESTAMP] = timestamp;
}

void Gen::set_max_time(float time){
	CHECK_GE(time, 0);
	_max_time = time;
}

void Gen::set_dt(float time){
	CHECK_GE(time, 0);
	_dt = time;
	static_cast<float *>(_conf->mutable_cpu_data())[Database::IDX_CONF_DT] = time;
}

float Gen::current_time(){
	return _current_time;
}

float Gen::dt(){
	return _dt;
}


Gen::mode_t Gen::current_mode(){
	return _current_mode;
}


}
