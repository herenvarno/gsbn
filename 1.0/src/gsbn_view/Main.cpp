#include "gsbn_view/Main.hpp"

INITIALIZE_EASYLOGGINGPP

using namespace gv;

int main(int argc, char **argv){
/*	
		SolverState solver_state;
		fstream input(argv[1], ios::in | ios::binary);
    if (!input) {
      LOG(FATAL) << "File not found, abort!";
    } else if (!solver_state.ParseFromIstream(&input)) {
    	LOG(FATAL) << "Parse file error, abort!";
    }
    
    Database db;
    db.init_copy(solver_state);

		LOG(INFO) << "spk:" << endl << db.table("spk")->dump();*/
		
	QApplication app(argc, argv);
/*	
	QUiLoader loader;
	QFile file("gsbn_viewer.ui");
	file.open(QFile::ReadOnly);
	QWidget *gv_mainwindow = loader.load(&file);
	file.close();
	*/
	Ui ui;
	ui.show();
	return app.exec();
}
