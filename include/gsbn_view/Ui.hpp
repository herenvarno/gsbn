#ifndef __GV_UI_HPP__
#define __GV_UI_HPP__

#include "gsbn_view/Common.hpp"

namespace gv{

class Ui : public QMainWindow{
	Q_OBJECT
public:
	Ui();
	~Ui();

private slots:
	void on_btn_open_clicked();
	void on_btn_show_clicked();

private:
	QPushButton *_btn_open;
	QLCDNumber *_lcd_timestamp;
	QSpinBox *_spin_hcu;
	QComboBox *_combo_param;
	QPushButton *_btn_show;
	QGraphicsView *_graph_f1;
	QTableView *_table_t2;
	
	Database *_db;
};

}


#endif
