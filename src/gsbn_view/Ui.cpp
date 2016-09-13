#include "gsbn_view/Ui.hpp"

namespace gv{

Ui::Ui() : _db(NULL){
    QWidget *widget = new QWidget;
    setCentralWidget(widget);
    QVBoxLayout *layout = new QVBoxLayout;
    layout->setMargin(5);
    widget->setLayout(layout);
    
    widget = new QWidget;
    layout->addWidget(widget);
    QHBoxLayout *top_hbox = new QHBoxLayout;
    widget->setLayout(top_hbox);
    
    _btn_open = new QPushButton("Open");
    top_hbox->addWidget(_btn_open);
    connect(_btn_open, SIGNAL(clicked()), SLOT(on_btn_open_clicked()));
    
    _lcd_timestamp = new QLCDNumber(7);
    _lcd_timestamp->setSegmentStyle(QLCDNumber::Flat);
    top_hbox->addWidget(_lcd_timestamp);
    
    top_hbox->addWidget(new QLabel("s"));
    
    _spin_hcu = new QSpinBox;
    top_hbox->addWidget(_spin_hcu);
    
    QStringList list;
    list
    	<< "conf"
    	<< "stim"
    	<< "tmp1" << "tmp2" << "tmp3"
    	<< "spk"
    	<< "j_array" << "epsc" << "sup"
    	<< "i_array" << "ij_mat" << "wij"
    	<< "pop" << "hcu" << "mcu" << "proj"
    	<< "hcu_slot" << "mcu_fanout"
    	<< "hcu_isp" << "hcu_osp"
    	<< "conn" << "conn0"
    	<< "rnd_uniform01" << "rnd_normal";
    _combo_param = new QComboBox;
    _combo_param->addItems(list);
    top_hbox->addWidget(_combo_param);
    
    _btn_show = new QPushButton("Show");
    top_hbox->addWidget(_btn_show);
    connect(_btn_show, SIGNAL(clicked()), SLOT(on_btn_show_clicked()));
    
    QTabWidget *tab = new QTabWidget();
    layout->addWidget(tab);
    
    QScrollArea *sa = new QScrollArea;
    tab->addTab(sa, "Table");
    
    _table_t2 = new QTableView;
    sa->setWidget(_table_t2);
    sa->setWidgetResizable(true);
    
    _graph_f1 = new QGraphicsView;
    tab->addTab(_graph_f1, "Figure");
  
}

Ui::~Ui(){
	if(_db){
		delete _db;
	}
}

void Ui::on_btn_open_clicked(){
	QString file = QFileDialog::getOpenFileName(this, "Open File", QDir::currentPath());

	if (file.isEmpty()) {
		return;
	}
	
	SolverState solver_state;
	fstream input(file.toStdString(), ios::in | ios::binary);
	if (!input) {
		QMessageBox::warning(this, "Warning", "File not found, abort!");
		return;
	} else if (!solver_state.ParseFromIstream(&input)) {
		QMessageBox::warning(this, "Warning", "Parse file error, abort!");
		return;
	}

	if(_db){
		delete _db;
	}
	_db = new Database;
	_db->init_copy(solver_state);
	float timestamp = static_cast<const float*>(_db->table("conf")->cpu_data(0))[Database::IDX_CONF_TIMESTAMP];
	_lcd_timestamp -> display(timestamp);
	_spin_hcu->setMaximum(_db->table("hcu")->height()-1);
	QMessageBox::information(this, "Info", "Load state success!");
	
}

void Ui::on_btn_show_clicked(){

	if(!_db){
		QMessageBox::warning(this, "Warning", "Database hasn't been initialized, load a state file first!");
		return;
	}
	
	int hcu_idx = _spin_hcu->value();
	QString param = _combo_param->currentText();
	
	QStandardItemModel *model = static_cast<QStandardItemModel*>(_table_t2->model());
	if(model){
		delete model;
	}
	
	Table *tab=_db->table(param.toStdString());
	if(!tab){
		return;
	}
	model = new QStandardItemModel(tab->rows(), tab->cols(), this);
	_table_t2->setModel(model);
	int h=tab->height();
	
	if(param=="spk"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("Spike")));
		for(int i=0; i<h; i++){
			const unsigned char *ptr = static_cast<const unsigned char*>(tab->cpu_data(i));
			unsigned char spk = ptr[Database::IDX_SPK_VALUE];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(int(spk)));
			model->setItem(i, 0, item);
		}
	}else if(param=="conf"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("timestamp")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("dt")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("prn")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("gain_mask")));
		model->setHorizontalHeaderItem(4, new QStandardItem(QString("plasticity")));
		model->setHorizontalHeaderItem(5, new QStandardItem(QString("stim")));
		for(int i=0; i<h; i++){
			const void *ptr = static_cast<const void*>(tab->cpu_data(i));
			float p0 = static_cast<const float*>(ptr)[Database::IDX_CONF_TIMESTAMP];
			float p1 = static_cast<const float*>(ptr)[Database::IDX_CONF_DT];
			float p2 = static_cast<const float*>(ptr)[Database::IDX_CONF_PRN];
			float p3 = static_cast<const float*>(ptr)[Database::IDX_CONF_GAIN_MASK];
			int p4 = static_cast<const int*>(ptr)[Database::IDX_CONF_PLASTICITY];
			int p5 = static_cast<const int*>(ptr)[Database::IDX_CONF_STIM];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
			item = new QStandardItem(QString::number(p4));
			model->setItem(i, 4, item);
			item = new QStandardItem(QString::number(p5));
			model->setItem(i, 5, item);
		}
	}
	else if(param=="stim"){
		for(int i=0; i<h; i++){
			const void *ptr = static_cast<const void*>(tab->cpu_data(i));
			QStandardItem *item = NULL;
			for(int j=0; j<tab->cols(); j++){
				float p0 = static_cast<const float*>(ptr)[j];
				item = new QStandardItem(QString::number(p0));
				model->setItem(i, j, item);
			}
		}
	}
	else if(param=="i_array"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("Pi")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("Ei")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("Zi")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("Ti")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_I_ARRAY_PI];
			float p1 = ptr[Database::IDX_I_ARRAY_EI];
			float p2 = ptr[Database::IDX_I_ARRAY_ZI];
			float p3 = ptr[Database::IDX_I_ARRAY_TI];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
		}
	}
	else if(param=="j_array"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("Pj")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("Ej")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("Zj")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("Bj")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_J_ARRAY_PJ];
			float p1 = ptr[Database::IDX_J_ARRAY_EJ];
			float p2 = ptr[Database::IDX_J_ARRAY_ZJ];
			float p3 = ptr[Database::IDX_J_ARRAY_BJ];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
		}
	}else if(param=="ij_mat"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("Pij")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("Eij")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("Zi2")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("Zj2")));
		model->setHorizontalHeaderItem(4, new QStandardItem(QString("Tij")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_IJ_MAT_PIJ];
			float p1 = ptr[Database::IDX_IJ_MAT_EIJ];
			float p2 = ptr[Database::IDX_IJ_MAT_ZI2];
			float p3 = ptr[Database::IDX_IJ_MAT_ZJ2];
			float p4 = ptr[Database::IDX_IJ_MAT_TIJ];
			
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
			item = new QStandardItem(QString::number(p4));
			model->setItem(i, 4, item);
		}
	}
	else if(param=="sup"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("dsup")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("act")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_SUP_DSUP];
			float p1 = ptr[Database::IDX_SUP_ACT];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
		}
	}
	else if(param=="epsc"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("epsc")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_EPSC_VALUE];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
		}
	}
	else if(param=="wij"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("wij")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_WIJ_VALUE];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
		}
	}
	else if(param=="conn"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("SRC MCU")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("DEST HCU")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("SUBPROJ")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("PROJ")));
		model->setHorizontalHeaderItem(4, new QStandardItem(QString("DELAY")));
		model->setHorizontalHeaderItem(5, new QStandardItem(QString("QUEUE")));
		model->setHorizontalHeaderItem(6, new QStandardItem(QString("TYPE")));
		model->setHorizontalHeaderItem(7, new QStandardItem(QString("IJ_MAT_INDEX")));
		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_CONN_SRC_MCU];
			int p1 = ptr[Database::IDX_CONN_DEST_HCU];
			int p2 = ptr[Database::IDX_CONN_SUBPROJ];
			int p3 = ptr[Database::IDX_CONN_PROJ];
			int p4 = ptr[Database::IDX_CONN_DELAY];
			int p5 = ptr[Database::IDX_CONN_QUEUE];
			int p6 = ptr[Database::IDX_CONN_TYPE];
			int p7 = ptr[Database::IDX_CONN_IJ_MAT_INDEX];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
			item = new QStandardItem(QString::number(p4));
			model->setItem(i, 4, item);
			item = new QStandardItem(QString::number(p5));
			model->setItem(i, 5, item);
			item = new QStandardItem(QString::number(p6));
			model->setItem(i, 6, item);
			item = new QStandardItem(QString::number(p7));
			model->setItem(i, 7, item);
		}
	}
	else if(param=="conn0"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("SRC MCU")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("DEST HCU")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("SUBPROJ")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("PROJ")));
		model->setHorizontalHeaderItem(4, new QStandardItem(QString("DELAY")));
		model->setHorizontalHeaderItem(5, new QStandardItem(QString("QUEUE")));
		model->setHorizontalHeaderItem(6, new QStandardItem(QString("TYPE")));
		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_CONN0_SRC_MCU];
			int p1 = ptr[Database::IDX_CONN0_DEST_HCU];
			int p2 = ptr[Database::IDX_CONN0_SUBPROJ];
			int p3 = ptr[Database::IDX_CONN0_PROJ];
			int p4 = ptr[Database::IDX_CONN0_DELAY];
			int p5 = ptr[Database::IDX_CONN0_QUEUE];
			int p6 = ptr[Database::IDX_CONN0_TYPE];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
			item = new QStandardItem(QString::number(p4));
			model->setItem(i, 4, item);
			item = new QStandardItem(QString::number(p5));
			model->setItem(i, 5, item);
			item = new QStandardItem(QString::number(p6));
			model->setItem(i, 6, item);
		}
	}
	else if(param=="tmp1"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("SPK MCU INDEX")));
		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_TMP1_MCU_IDX];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
		}
	}
	else if(param=="tmp2"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("CONN INDEX")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("SRC MCU")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("DEST HCU")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("SUBPROJ")));
		model->setHorizontalHeaderItem(4, new QStandardItem(QString("PROJ")));
		model->setHorizontalHeaderItem(5, new QStandardItem(QString("IJ_MAT_INDEX")));
		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_TMP2_CONN];
			int p1 = ptr[Database::IDX_TMP2_SRC_MCU];
			int p2 = ptr[Database::IDX_TMP2_DEST_HCU];
			int p3 = ptr[Database::IDX_TMP2_SUBPROJ];
			int p4 = ptr[Database::IDX_TMP2_PROJ];
			int p5 = ptr[Database::IDX_TMP2_IJ_MAT_INDEX];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
			item = new QStandardItem(QString::number(p4));
			model->setItem(i, 4, item);
			item = new QStandardItem(QString::number(p5));
			model->setItem(i, 5, item);
		}
	}
	else if(param=="tmp3"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("CONN INDEX")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("DEST_HCU")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("IJ_MAT")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("PI_INIT")));
		model->setHorizontalHeaderItem(4, new QStandardItem(QString("PIJ_INIT")));
		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_TMP3_CONN];
			int p1 = ptr[Database::IDX_TMP3_DEST_HCU];
			int p2 = ptr[Database::IDX_TMP3_IJ_MAT_IDX];
			const float *ptr0 = static_cast<const float*>(tab->cpu_data(i));
			float p3 = ptr0[Database::IDX_TMP3_PI_INIT];
			float p4 = ptr0[Database::IDX_TMP3_PIJ_INIT];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
			item = new QStandardItem(QString::number(p4));
			model->setItem(i, 4, item);
		}
	}
	else if(param=="proj"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("SRC POP")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("DEST POP")));
		model->setHorizontalHeaderItem(2, new QStandardItem(QString("MCU_NUM")));
		model->setHorizontalHeaderItem(3, new QStandardItem(QString("TAUZIDT")));
		model->setHorizontalHeaderItem(4, new QStandardItem(QString("TAUZJDT")));
		model->setHorizontalHeaderItem(5, new QStandardItem(QString("TAUEDT")));
		model->setHorizontalHeaderItem(6, new QStandardItem(QString("TAUPDT")));
		model->setHorizontalHeaderItem(7, new QStandardItem(QString("EPS")));
		model->setHorizontalHeaderItem(8, new QStandardItem(QString("EPS2")));
		model->setHorizontalHeaderItem(9, new QStandardItem(QString("KFTI")));
		model->setHorizontalHeaderItem(10, new QStandardItem(QString("KFTJ")));
		model->setHorizontalHeaderItem(11, new QStandardItem(QString("BGAIN")));
		model->setHorizontalHeaderItem(12, new QStandardItem(QString("WGAIN")));
		model->setHorizontalHeaderItem(13, new QStandardItem(QString("PI0")));
		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_PROJ_SRC_POP];
			int p1 = ptr[Database::IDX_PROJ_DEST_POP];
			int p2 = ptr[Database::IDX_PROJ_MCU_NUM];
			const float *ptr0 = static_cast<const float*>(tab->cpu_data(i));
			float p3 = ptr0[Database::IDX_PROJ_TAUZIDT];
			float p4 = ptr0[Database::IDX_PROJ_TAUZJDT];
			float p5 = ptr0[Database::IDX_PROJ_TAUEDT];
			float p6 = ptr0[Database::IDX_PROJ_TAUPDT];
			float p7 = ptr0[Database::IDX_PROJ_EPS];
			float p8 = ptr0[Database::IDX_PROJ_EPS2];
			float p9 = ptr0[Database::IDX_PROJ_KFTI];
			float p10 = ptr0[Database::IDX_PROJ_KFTJ];
			float p11 = ptr0[Database::IDX_PROJ_BGAIN];
			float p12 = ptr0[Database::IDX_PROJ_WGAIN];
			float p13 = ptr0[Database::IDX_PROJ_PI0];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
			item = new QStandardItem(QString::number(p2));
			model->setItem(i, 2, item);
			item = new QStandardItem(QString::number(p3));
			model->setItem(i, 3, item);
			item = new QStandardItem(QString::number(p4));
			model->setItem(i, 4, item);
			item = new QStandardItem(QString::number(p5));
			model->setItem(i, 5, item);
			item = new QStandardItem(QString::number(p6));
			model->setItem(i, 6, item);
			item = new QStandardItem(QString::number(p7));
			model->setItem(i, 7, item);
			item = new QStandardItem(QString::number(p8));
			model->setItem(i, 8, item);
			item = new QStandardItem(QString::number(p9));
			model->setItem(i, 9, item);
			item = new QStandardItem(QString::number(p10));
			model->setItem(i, 10, item);
			item = new QStandardItem(QString::number(p11));
			model->setItem(i, 11, item);
			item = new QStandardItem(QString::number(p12));
			model->setItem(i, 12, item);
			item = new QStandardItem(QString::number(p13));
			model->setItem(i, 13, item);
		}
	}
	else if(param=="pop"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("HCU IDX")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("HCU NUM")));

		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_POP_HCU_INDEX];
			int p1 = ptr[Database::IDX_POP_HCU_NUM];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
		}
	}
	else if(param=="mcu"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("J ARRAY IDX")));
		model->setHorizontalHeaderItem(1, new QStandardItem(QString("J ARRAY NUM")));

		for(int i=0; i<h; i++){
			const int *ptr = static_cast<const int*>(tab->cpu_data(i));
			int p0 = ptr[Database::IDX_MCU_J_ARRAY_INDEX];
			int p1 = ptr[Database::IDX_MCU_J_ARRAY_NUM];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
			item = new QStandardItem(QString::number(p1));
			model->setItem(i, 1, item);
		}
	}else if(param=="rnd_uniform01"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("UNIFORM01")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_RND_UNIFORM01_VALUE];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
		}
	}else if(param=="rnd_normal"){
		model->setHorizontalHeaderItem(0, new QStandardItem(QString("NORMAL")));
		for(int i=0; i<h; i++){
			const float *ptr = static_cast<const float*>(tab->cpu_data(i));
			float p0 = ptr[Database::IDX_RND_NORMAL_VALUE];
			QStandardItem *item = NULL;
			item = new QStandardItem(QString::number(p0));
			model->setItem(i, 0, item);
		}
	}
	
}

}
