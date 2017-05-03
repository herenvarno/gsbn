#include "gsbn/procedures/ProcUpdMulti/Proj.hpp"


#ifndef CPU_ONLY

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

using namespace thrust;
using namespace thrust::placeholders;

namespace gsbn{
namespace proc_upd_multi{

__global__ void update_all_kernel_gpu(
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
	float eps2
){
	int i=blockIdx.y*gridDim.x+blockIdx.x;
	int j=threadIdx.x;

	__shared__ float sh_pi;
	__shared__ float sh_zi;
	__shared__ int sh_ti;
	if(j==0){
		float pi = ptr_pi[i];
		float zi = ptr_zi[i];
		sh_zi = zi;
		int ti = ptr_ti[i];
		sh_ti = ti;
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
		sh_pi = pi;
	}
	__syncthreads();
	
	int index = i*dim_mcu+j;
	
	float zi2 = sh_zi;
	int tij = sh_ti;
	
	if(simstep-tij>0){
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zj2 = ptr_zj2[index];
		
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

		// update wij and mark inactive connection
		float wij;
		if(kp){
			__shared__ int active_flag;
			if(j==0){
				active_flag = 0;
			}
			__syncthreads();
			float pi = sh_pi;
			float pj = ptr_pj[i/dim_conn*dim_mcu + j];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
			
			if(wij>-5){
				active_flag = 1;
			}
			__syncthreads();
			if(j==0){
				if(!active_flag){
					ptr_ti[i] = -1;
				}
			}
		}
	}
}
__global__ void update_jxx_kernel_gpu(
	int n,
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
	CUDA_KERNEL_LOOP(idx, n){
		float pj = ptr_pj[idx];
		float ej = ptr_ej[idx];
		float zj = ptr_zj[idx];
		int8_t sj = ptr_sj[idx];
		
		ptr_epsc[idx] *= (1-kepsc);
		
		pj += (ej - pj)*kp;
		ej += (zj - ej)*ke;
		zj *= (1-kzj);
		if(sj>0){
			zj += kftj;
		}

		if(kp){
			float bj = bgain * log(pj + eps);
			ptr_bj[idx]=bj;
		}
		ptr_pj[idx] = pj;
		ptr_ej[idx] = ej;
		ptr_zj[idx] = zj;
	}
}

__global__ void update_row_kernel_gpu(
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
	float eps2
){
	int i = blockIdx.x;
	int j = threadIdx.x;
	int row = ptr_ssi[i];
	int col = j;
	int index = row*dim_mcu+col;

	__shared__ float sh_pi;
	__shared__ float sh_zi;
	__shared__ int sh_ti;

	if(j==0){
		float pi = ptr_pi[row];
		float zi = ptr_zi[row];
		sh_zi = zi;
		int ti = ptr_ti[row];
		sh_ti = ti;
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_zi[row] += kfti;
			ptr_ti[row] = simstep;
		}else{
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
		sh_pi = pi;
	}
	
	__syncthreads();
	
	float zi2 = sh_zi;
	int tij = sh_ti;
	if(simstep - tij >0){
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zj2 = ptr_zj2[index];

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
		if(kp){
			float pi = sh_pi;
			float pj = ptr_pj[i/dim_conn*dim_mcu + j];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
			
			
		}
		
		float wij;
		int idx_hcu = row / dim_conn;
		int idx_mcu = idx_hcu * dim_mcu + j;
		if(kp){
			__shared__ int active_flag;
			if(j==0){
				active_flag = 0;
			}
			__syncthreads();
			float pi = sh_pi;
			float pj = ptr_pj[idx_mcu];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
			if(wij>-5){
				active_flag = 1;
			}
			__syncthreads();
			if(j==0){
				if(!active_flag){
					ptr_ti[i] = -1;
				}
			}
		}else{
			wij = ptr_wij[index];
		}
		atomicAdd(&ptr_epsc[idx_mcu], wij);
	}
}

void Proj::update_all_gpu(){
	int simstep;
	float prn;
	float old_prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("prn", prn));
	CHECK(_glv.getf("old-prn", old_prn));
	if(old_prn!=prn){
		float *ptr_pi = _pi->mutable_gpu_data();
		float *ptr_ei = _ei->mutable_gpu_data();
		float *ptr_zi = _zi->mutable_gpu_data();
		int *ptr_ti = _ti->mutable_gpu_data();
		const float *ptr_pj = _pj->gpu_data();
		float *ptr_pij = _pij->mutable_gpu_data();
		float *ptr_eij = _eij->mutable_gpu_data();
		float *ptr_zj2 = _zj2->mutable_gpu_data();
		float *ptr_wij = _wij->mutable_gpu_data();
		const int8_t *ptr_sj = _sj->gpu_data();
		
		const dim3 GRID_SIZE(_dim_conn, _dim_hcu);
		CUDA_CHECK(cudaSetDevice(_device));
		update_all_kernel_gpu<<<GRID_SIZE, _dim_mcu, 0, _stream>>>(
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
			_eps2
		);
		CUDA_POST_KERNEL_CHECK;
	}
}

void Proj::update_jxx_gpu(){
	int simstep;
	float prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("prn", prn));
	float *ptr_pj = _pj->mutable_gpu_data();
	float *ptr_ej = _ej->mutable_gpu_data();
	float *ptr_zj = _zj->mutable_gpu_data();
	float *ptr_bj = _bj->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int8_t *ptr_sj = _sj->gpu_data()+(simstep%_spike_buffer_size)*_dim_hcu*_dim_mcu;
	
	CUDA_CHECK(cudaSetDevice(_device));
	update_jxx_kernel_gpu<<<GSBN_GET_BLOCKS(_dim_hcu*_dim_mcu), GSBN_GET_THREADS(_dim_hcu*_dim_mcu), 0, _stream>>>(
		_dim_hcu*_dim_mcu,
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
	CUDA_POST_KERNEL_CHECK;
}

__global__ void update_que_kernel_gpu(
	int n,
	const int *ptr_ii,
	const int *ptr_di,
	const int8_t *ptr_si,
	int *ptr_qi,
	int *ptr_siq
){
	CUDA_KERNEL_LOOP(i, n){
		int ii=ptr_ii[i];
		if(ii>=0){
			int32_t qi = ptr_qi[i];
			
			ptr_siq[i] = int(qi & 0x01);
			
			int8_t spk = ptr_si[ii];
			if(spk>0){
				qi |= (0x01 << ptr_di[i]);
			}
			qi >>= 1;
			ptr_qi[i]=qi;
		}
	}
}

void Proj::update_row_gpu(){
	int active_row_num = _ssi->size();
	if(active_row_num<=0){
		return;
	}

	int simstep;
	float prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("prn", prn));

	float *ptr_pi = _pi->mutable_gpu_data();
	float *ptr_ei = _ei->mutable_gpu_data();
	float *ptr_zi = _zi->mutable_gpu_data();
	int *ptr_ti = _ti->mutable_gpu_data();
	const float *ptr_pj = _pj->gpu_data();
	float *ptr_pij = _pij->mutable_gpu_data();
	float *ptr_eij = _eij->mutable_gpu_data();
	float *ptr_zj2 = _zj2->mutable_gpu_data();
	float *ptr_wij = _wij->mutable_gpu_data();
	float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int *ptr_siq = _siq->gpu_data();
	const int8_t *ptr_sj = _sj->gpu_data();

	const int *ptr_ssi = _ssi->gpu_data();
	
	CUDA_CHECK(cudaSetDevice(_device));
	update_row_kernel_gpu<<<active_row_num, _dim_mcu, 0, _stream>>>(
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
		_eps2
	);
	CUDA_POST_KERNEL_CHECK;
}

void Proj::update_ssi_gpu(){
	CONST_DEVICE_VECTOR(int, *v_siq) = _siq->gpu_vector();
	_ssi->resize(v_siq->size());
	DEVICE_VECTOR(int, *v_ssi) = _ssi->mutable_gpu_vector();
	auto it = copy_if(
		thrust::cuda::par.on(_stream),
		make_counting_iterator<int>(0),
		make_counting_iterator<int>(v_siq->size()),
		v_siq->begin(),
		v_ssi->begin(),
		_1>0);
	_ssi->resize(thrust::distance(v_ssi->begin(), it));
}

void Proj::update_ssj_gpu(){
	int simstep;
	CHECK(_glv.geti("simstep", simstep));
	int offset=(simstep%_spike_buffer_size)*_dim_hcu*_dim_mcu;
	CONST_DEVICE_VECTOR(int8_t, *v_sj) = _sj->gpu_vector();
	_ssj->resize(_dim_hcu*_dim_mcu);
	DEVICE_VECTOR(int, *v_ssj) = _ssj->mutable_gpu_vector();
	auto it = copy_if(
		thrust::cuda::par.on(_stream),
		make_counting_iterator<int>(0),
		make_counting_iterator<int>(_dim_hcu*_dim_mcu),
		v_sj->begin()+offset,
		v_ssj->begin(),
		_1>0);
	_ssj->resize(thrust::distance(v_ssj->begin(), it));
}

void Proj::update_que_gpu(){
	const int *ptr_ii = _ii->gpu_data();
	const int *ptr_di = _di->gpu_data();
	const int8_t *ptr_si = _si->gpu_data();
	int *ptr_qi = _qi->mutable_gpu_data();
	int *ptr_siq = _siq->mutable_gpu_data();
	
	update_que_kernel_gpu<<<GSBN_GET_BLOCKS(_dim_hcu* _dim_conn), GSBN_GET_THREADS(_dim_hcu* _dim_conn)>>>(
		_dim_hcu * _dim_conn,
		ptr_ii,
		ptr_di,
		ptr_si,
		ptr_qi,
		ptr_siq
	);
	CUDA_POST_KERNEL_CHECK;
}

}
}

#endif

