#include "gsbn/Solver.hpp"
#include <algorithm>
#include <string>
#include <dirent.h>

using namespace gsbn;
using namespace std;

INITIALIZE_EASYLOGGINGPP

bool has_suffix(const string& s, const string& suffix)
{
    return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());    
}


void dump_spk(Database&db, string o_path);
void dump_spk_short(Database&db, string o_path);
void dump_stim(Database&db, string o_path);
void dump_ij(Database&db, string o_path);
void dump_i(Database&db, string o_path);
void dump_j(Database&db, string o_path);

int main(int argc, char *argv[]){

	char *i_dir_ptr = NULL;
	char *o_dir_ptr = NULL;
	int c;

	while ((c = getopt (argc, argv, "i:o:")) != -1){
		switch (c){
		case 'i':
			i_dir_ptr = optarg;
			break;
		case 'o':
			o_dir_ptr = optarg;
			break;
		case '?':
		default:
			LOG(FATAL) << "Arguments wrong, abort!";
		}
	}
	
  for (int index = optind; index < argc; index++){
		LOG(WARNING) << "Non-option argument " << argv[index];
	}
	
	CHECK(i_dir_ptr && o_dir_ptr) << "Incompleted arguments!";
	
	string i_dir(i_dir_ptr);
	string o_dir(o_dir_ptr);
	
	i_dir = i_dir + '/';
	o_dir = o_dir + '/';
	
	
	
	DIR *dir = opendir(i_dir.c_str());
	CHECK(dir) << "Invalid input directory!";

	dirent *entry;
	vector<string> file_list;
	while((entry = readdir(dir))!=NULL){
		if(has_suffix(entry->d_name, ".bin")){
			file_list.push_back(i_dir+entry->d_name);
		}
	}
	closedir(dir);
	
	for(vector<string>::iterator it=file_list.begin(); it!=file_list.end(); it++){
		struct stat info;
		if(stat( o_dir.c_str(), &info )!=0 || !(info.st_mode & S_IFDIR)){
			LOG(WARNING) << "Directory does not exist! Create one!";
			string cmd="mkdir -p "+o_dir;
			if(system(cmd.c_str())!=0){
				LOG(FATAL) << "Cannot create directory for state records! Aboart!";
			}
		}
		
		SolverState solver_state;
		fstream input(*it, ios::in | ios::binary);
		if (!input) {
			LOG(FATAL) << "File not found, abort!";
		} else if (!solver_state.ParseFromIstream(&input)) {
			LOG(FATAL) << "Parse file error, abort!";
		}
		
		Database db;
		db.init_copy(solver_state);
  
		float timestamp = solver_state.timestamp();
		LOG(INFO) << "Processing timestamp: " << timestamp;
		string o_path = o_dir + to_string(timestamp) + "/";
		string cmd="mkdir -p "+o_path;
		if(system(cmd.c_str())!=0){
			LOG(FATAL) << "Cannot create directory for state records! Abort!";
		}
		
		//dump_spk(db, o_path);
		dump_spk_short(db, o_path);
		//dump_stim(db, o_path);
		//dump_ij(db, o_path);
		//dump_i(db, o_path);
		//dump_j(db, o_path);
	}
	
	return 0;
}


void dump_spk(Database&db, string o_path){
	string target_file = o_path + "spk.txt";
	ofstream fd;
	fd.open(target_file, ios::out);
	
	Table *t = db.table("spk");
	int h = t->height();
	for(int i=0; i<h; i++){
		unsigned char s = static_cast<const unsigned char *>(t->cpu_data(i))[Database::IDX_SPK_VALUE];
		fd << int(s) << endl;
	}
}

void dump_spk_short(Database&db, string o_path){
	string target_file = o_path + "spk_short.txt";
	ofstream fd;
	fd.open(target_file, ios::out);
	
	Table *t = db.table("tmp1");
	int h = t->height();
	for(int i=0; i<h; i++){
		int s = static_cast<const int *>(t->cpu_data(i))[Database::IDX_TMP1_MCU_IDX];
		fd << s << endl;
	}
}

void dump_stim(Database&db, string o_path){
	string target_file = o_path + "stim.txt";
	ofstream fd;
	fd.open(target_file, ios::out);
	
	int stim_idx = *static_cast<const int *>(db.table("conf")->cpu_data(0, Database::IDX_CONF_STIM));
	
	Table *t = db.table("stim");
	int c = t->cols();
	for(int i=0; i<c; i++){
		float s = *static_cast<const float *>(t->cpu_data(stim_idx, i));
		fd << s << endl;
	}
}

void dump_ij(Database&db, string o_path){
	string target_file = o_path + "ij.txt";
	ofstream fd;
	fd.open(target_file, ios::out);
	
	Table *t = db.table("conn");
	Table *t1 = db.table("ij_mat");
	Table *t2 = db.table("hcu");
	Table *t3 = db.table("wij");
	int h = t->height();
	for(int i=0; i<h; i++){
		int mcu = *static_cast<const int *>(t->cpu_data(i, Database::IDX_CONN_SRC_MCU));
		int hcu = *static_cast<const int *>(t->cpu_data(i, Database::IDX_CONN_DEST_HCU));
		int proj = *static_cast<const int *>(t->cpu_data(i, Database::IDX_CONN_PROJ));
		int subproj = *static_cast<const int *>(t->cpu_data(i, Database::IDX_CONN_SUBPROJ));
		int ij_mat_idx = *static_cast<const int *>(t->cpu_data(i, Database::IDX_CONN_IJ_MAT_INDEX));
		int mcu_num = *static_cast<const int *>(t2->cpu_data(hcu, Database::IDX_HCU_MCU_NUM));
		int first_mcu = *static_cast<const int *>(t2->cpu_data(hcu, Database::IDX_HCU_MCU_INDEX));
		for(int j=0; j<mcu_num; j++){
			float pij = *static_cast<const float *>(t1->cpu_data(ij_mat_idx+j, Database::IDX_IJ_MAT_PIJ));
			float eij = *static_cast<const float *>(t1->cpu_data(ij_mat_idx+j, Database::IDX_IJ_MAT_PIJ));
			float zi2 = *static_cast<const float *>(t1->cpu_data(ij_mat_idx+j, Database::IDX_IJ_MAT_ZI2));
			float zj2 = *static_cast<const float *>(t1->cpu_data(ij_mat_idx+j, Database::IDX_IJ_MAT_ZJ2));
			float tij = *static_cast<const float *>(t1->cpu_data(ij_mat_idx+j, Database::IDX_IJ_MAT_TIJ));
			float wij = *static_cast<const float *>(t3->cpu_data(ij_mat_idx+j, Database::IDX_WIJ_VALUE));
			fd << mcu << " " << first_mcu+j << " ";
			fd << pij << " " << eij << " " << zi2 << " " << zj2 << " " << tij <<  " " << wij;
			fd << endl;
		}
	}
}

void dump_i(Database&db, string o_path){
	string target_file = o_path + "i.txt";
	ofstream fd;
	fd.open(target_file, ios::out);
	
	Table *t = db.table("conn");
	Table *t1 = db.table("i_array");
	int h = t->height();
	for(int i=0; i<h; i++){
		int mcu = *static_cast<const int *>(t->cpu_data(i, Database::IDX_CONN_SRC_MCU));
		int hcu = *static_cast<const int *>(t->cpu_data(i, Database::IDX_CONN_DEST_HCU));
		float pi = *static_cast<const float *>(t1->cpu_data(i, Database::IDX_I_ARRAY_PI));
		float ei = *static_cast<const float *>(t1->cpu_data(i, Database::IDX_I_ARRAY_EI));
		float zi = *static_cast<const float *>(t1->cpu_data(i, Database::IDX_I_ARRAY_ZI));
		fd << mcu << " "<< hcu << " " << pi << " " << ei << " " << zi << endl;
	}
}

// FIXME
void dump_j(Database&db, string o_path){
	string target_file = o_path + "j.txt";
	ofstream fd;
	fd.open(target_file, ios::out);
	
	Table *t = db.table("mcu");
	Table *t1 = db.table("j_array");
	Table *t2 = db.table("epsc");
	int h = t->height();
	for(int i=0; i<h; i++){
		int mcu = i;
		float pj = *static_cast<const float *>(t1->cpu_data(i, Database::IDX_J_ARRAY_PJ));
		float ej = *static_cast<const float *>(t1->cpu_data(i, Database::IDX_J_ARRAY_EJ));
		float zj = *static_cast<const float *>(t1->cpu_data(i, Database::IDX_J_ARRAY_ZJ));
		float bj = *static_cast<const float *>(t1->cpu_data(i, Database::IDX_J_ARRAY_BJ));
		float epsc = *static_cast<const float *>(t2->cpu_data(i, Database::IDX_EPSC_VALUE));
		fd << mcu << " " << pj << " " << ej << " " << zj << " " << bj << " " << epsc << endl;
	}
}

