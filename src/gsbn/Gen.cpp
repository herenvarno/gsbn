#include "gsbn/Gen.hpp"

namespace gsbn{

Gen::Gen() : _current_time(0), _current_mode(Gen::NOP), _current_stim(-1), _max_time(-1){
}

void Gen::init(Database& db){
	CHECK(_mode = db.table("mode"));
	CHECK(_stim = db.table("stim"));
	_max_time = static_cast<const int *>(_mode->cpu_data(_mode->height()-1))[Database::IDX_MODE_END_TIME];
}
	
void Gen::update(){
	_current_time++;
	if(_current_time>_max_time){
		_current_mode = END;
		_current_stim = -1;
		return;
	}
	
	_current_stim = -1;
	int r=_mode->rows();
	int l=0, h=r-1;
	int m=(l+h)/2;
	int mode=0, stim=-1;
	while(1){
		int bt0, et0;
		const int *ptr = static_cast<const int*>(_mode->cpu_data(m));
		bt0 = ptr[Database::IDX_MODE_BEGIN_TIME];
		et0 = ptr[Database::IDX_MODE_END_TIME];
		if(_current_time >= bt0 && _current_time <= et0){
			mode = ptr[Database::IDX_MODE_TYPE];
			stim = ptr[Database::IDX_MODE_STIM];
			break;
		}else if(_current_time < bt0){
			h = m;
			m = (h+l)/2;
			if(h == l){
				mode = 3;
				break;
			}
		}else{
			l = m;
			m = (h+1)/2;
			if(h == l){
				mode = 3;
				break;
			}
		}
	}
	switch(mode){
	case 0:
		_current_mode = LEARN;
		_current_stim = stim;
		break;
	case 1:
		_current_mode = RECALL;
		break;
	default:
		_current_mode = NOP;
	}
}

void Gen::set_current_time(int timestamp){
	CHECK_GE(timestamp, 0);
	_current_time = timestamp;
}

void Gen::set_max_time(int time){
	CHECK_GE(time, 0);
	_max_time = time;
}

int Gen::current_time(){
	return _current_time;
}

Gen::mode_t Gen::current_mode(){
	return _current_mode;
}

int Gen::current_stim(){
	return _current_stim;
}

}
