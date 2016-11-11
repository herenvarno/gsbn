#include "gsbn/procedures/ProcNetBatch/Proj.hpp"

#ifndef CPU_ONLY

namespace gsbn{
namespace proc_net_batch{

__global__ void update_full_kernel_gpu(
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
	int i=blockIdx.x;
	int j=threadIdx.x;
	
	__shared__ float sh_pi;
	if(j==0){
		float pi = ptr_pi[i];
		float zi = ptr_zi[i];
		int ti = ptr_ti[i];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_ti[i]=simstep;
		}else{
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
	
	int tij = ptr_tij[index];
	float zi2 = ptr_zi2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_tij[index]=simstep;
	}else{
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zj2 = ptr_zj2[index];
	
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
			 	
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi2;
		ptr_zj2[index] = zj2;
		ptr_tij[index] = tij;
			
		// update wij and epsc
		float wij;
		if(kp){
			float pi = sh_pi;
			float pj = ptr_pj[i/dim_conn*dim_mcu + j];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
		}
	}
}

__global__ void update_j_kernel_gpu(
	int n,
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
	CUDA_KERNEL_LOOP(idx, n){
		float pj = ptr_pj[idx];
		float ej = ptr_ej[idx];
		float zj = ptr_zj[idx];
		int sj = ptr_sj[idx];
		
		ptr_epsc[idx] *= (1-kzi);

		if(kp){
			float bj = bgain * log(pj + eps);
			ptr_bj[idx]=bj;
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
}

void Proj::update_j_gpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];
	float *ptr_pj = _pj->mutable_gpu_data();
	float *ptr_ej = _ej->mutable_gpu_data();
	float *ptr_zj = _zj->mutable_gpu_data();
	float *ptr_bj = _bj->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int *ptr_sj = _sj->gpu_data();

	update_j_kernel_gpu<<<GSBN_GET_BLOCKS(_dim_hcu*_dim_mcu), GSBN_GET_THREADS(_dim_hcu*_dim_mcu), 0, _stream>>>(
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
		_kftj,
		_bgain,
		_eps
	);
	CUDA_POST_KERNEL_CHECK;
}

}
}

#endif
