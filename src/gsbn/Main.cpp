#include "gsbn/Main.hpp"

using namespace gsbn;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
	Spike sp(100, 1);
	sp.append(200, 7);
	LOG(INFO) << "Table Spike:" << endl << sp.dump();
	
	Recorder rcd;
	rcd.set_directory("states_dir/");
	rcd.set_timestamp(1);
	rcd.set_freq(1);
	rcd.append_table(sp);
	rcd.record();
	
	Spike sp0(500, 9);
	LOG(INFO) << "Table Spike:" << endl << sp0.dump();
	
	SolverState st;
	fstream input("states_dir/SolverState_1.bin", ios::in | ios::binary);
  if (!st.ParseFromIstream(&input)) {
    cerr << "Failed to parse address book." << endl;
    return -1;
  }
  
  TableState tab_st=st.table_state(0);
  sp0.set_state(tab_st);
  LOG(INFO) << "Table Spike:" << endl << sp0.dump();
	
	
	
	
	return 0;
}
