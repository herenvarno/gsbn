#!/usr/bin/env python2

from __future__ import unicode_literals
import sys
import os
import random
from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

color_map=[]

lines = []
with open("colors.map") as f:
	lines = f.read().split()

if(len(lines)!=int(lines[0])+1):
	print("color map error!")
	
color_map = [int(x) for x in lines[1:]]


class MyMplCanvas(FigureCanvas):
	"""Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
	
	def __init__(self, parent=None, width=10, height=10, dpi=5):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = fig.add_subplot(111)
		
		self.compute_initial_figure()
		
		FigureCanvas.__init__(self, fig)
		self.setParent(parent)
		
		FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		fig.tight_layout()
	
	def compute_initial_figure(self):
		pass
	
class MyDynamicMplCanvas(MyMplCanvas):
	"""A canvas that updates itself every second with a new plot."""
	
	def __init__(self, *args, **kwargs):
		MyMplCanvas.__init__(self, *args, **kwargs)
		self.dim_hcu=0
		self.dim_mcu=0
		self.scale=1
	
	def compute_initial_figure(self):
		H = 20
		W = 20
		C = 2
		mat = np.zeros([H, W, 3], dtype=np.uint8)
		self.cax = self.axes.imshow(mat, interpolation='nearest')
		self.axes.set_title("t=0 ms")
	
	def show_pic(self, data, cursor, window):
		global color_map
		H = 20
		W = 20
		C = 2
		mat = np.zeros([H, W, 3], dtype=np.uint8)
		cnt = np.zeros([H, W, C, 40], dtype=np.uint8)
		
		bh = cursor
		bl = cursor - window + 1
		if bl<0:
			bl=0
		
		for i in range(len(data)):
			d = data[i]
			if d[0]<bl:
				continue
			if d[0]>bh:
				break
			
			for idx in d[1]:
				if(idx>=self.dim_hcu*self.dim_mcu or idx<0):
					continue;
				hcu_idx = idx // 40
				h = (hcu_idx//C)//W
				w = (hcu_idx//C)% W
				c = hcu_idx%C
				cnt[h][w][c][idx%40] += 1
		
		for h in range(H):
			for w in range(W):
				c_idx=0
				for c in range(C):
					c_idx = c_idx * 40 + np.argmax(cnt[h][w][C-c-1])
				if c_idx >= len(color_map):
					c_idx = -1
				color = color_map[c_idx]
				for x in range(3):
					mat[h][w][x] = (color >> 8*(2-x)) & 0xff
					
		mat = mat.astype(np.uint8)
		self.axes.cla()
		self.cax = self.axes.imshow(mat, interpolation='nearest')
		self.axes.set_title("t="+str(cursor)+" ms")
		self.draw()

class ApplicationWindow(QtGui.QMainWindow):
	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.btnFClicked)
		
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		self.setWindowTitle("application main window")
		
		self.file_menu = QtGui.QMenu('&File', self)
		self.file_menu.addAction('&Open', self.fileOpen, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
		self.file_menu.addAction('&Save', self.fileSave, QtCore.Qt.CTRL + QtCore.Qt.Key_S)
		self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
		self.menuBar().addMenu(self.file_menu)
		
		self.help_menu = QtGui.QMenu('&Help', self)
		self.menuBar().addSeparator()
		self.menuBar().addMenu(self.help_menu)
		
		self.help_menu.addAction('&About', self.about)
		
		self.menuBar().setNativeMenuBar(False)
		
		self.main_widget = QtGui.QWidget(self)
		
		l = QtGui.QVBoxLayout(self.main_widget)
		self.dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
		hbox = QtGui.QHBoxLayout()
		self.btn_b = QtGui.QPushButton(self.style().standardIcon(QtGui.QStyle.SP_MediaSeekBackward), "")
		self.btn_r = QtGui.QPushButton(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay), "")
		self.btn_r.setCheckable(True)
		self.btn_f = QtGui.QPushButton(self.style().standardIcon(QtGui.QStyle.SP_MediaSeekForward), "")
		self.bar = QtGui.QSlider(QtCore.Qt.Horizontal,self)
		self.label_t = QtGui.QLabel("0/0")
		self.btn_r.clicked.connect(self.btnRToggled)
		self.btn_b.clicked.connect(self.btnBClicked)
		self.btn_f.clicked.connect(self.btnFClicked)
		self.bar.valueChanged.connect(self.barValueChanged)
		self.btn_b.setEnabled(False)
		self.btn_f.setEnabled(False)
		self.btn_r.setEnabled(False)
		self.bar.setEnabled(False)
		hbox.addWidget(self.btn_b)
		hbox.addWidget(self.btn_r)
		hbox.addWidget(self.btn_f)
		hbox.addWidget(self.bar)
		hbox.addWidget(self.label_t)
		l.addWidget(self.dc)
		l.addLayout(hbox)
		
		self.dc.dim_hcu=0
		self.dc.dim_mcu=0
		
		self.main_widget.setFocus()
		self.setCentralWidget(self.main_widget)
		
		self.statusBar().showMessage("All hail matplotlib!", 2000)
	
	def fileOpen(self):
		self.statusBar().showMessage("Open file ...")
		filename = QtGui.QFileDialog.getOpenFileName(self, 'Open spike file ...', filter="Spike recording file (*.csv)")
		if filename:
			data=[]
			with open(filename, "r") as f:
				lines = f.read().splitlines()
			
			dim_flag=False
			for l in lines:
				var_list = l.split(",")
				if(len(var_list)<2):
					continue
				
				if not dim_flag:
					if(len(var_list)<4):
						continue
					self.dc.dim_hcu=int(var_list[0])
					self.dc.dim_mcu=int(var_list[1])
					self.dc.scale = (1.0/int(var_list[3])/float(var_list[2]))
					dim_flag=True
					continue
				
				coor=[]
				mcu_count = len(var_list)-1
				for i in range(mcu_count): 
					idx = int(var_list[1+i])
					coor.append(idx)
				data.append((int(var_list[0]), coor))
			
			print(len(data))
			self.data=data
			
			self.bar.setMaximum(data[len(data)-1][0])
			self.bar.triggerAction(QtGui.QAbstractSlider.SliderToMinimum)
			
			self.btn_b.setEnabled(True)
			self.btn_f.setEnabled(True)
			self.btn_r.setEnabled(True)
			self.bar.setEnabled(True)
			
			self.update_all()
			self.statusBar().showMessage("Spike file \"" + filename + "\" loaded!", 2000);
		else:
			self.statusBar().showMessage("Quit open file ...", 2000);
	
	def fileSave(self):
		global color_map
		self.statusBar().showMessage("Save to video ...")
		filename = QtGui.QFileDialog.getSaveFileName(self, 'Save to video file ...', filter="Video file (*.mp4)")
		if filename:
			start,ok = QtGui.QInputDialog.getInt(self,"Start timestamp","Start timestamp")
			if ok:
				end,ok = QtGui.QInputDialog.getInt(self,"End timestamp","End timestamp")
				if ok:
					end = end+1
					data = self.data
					dim_hcu = self.dc.dim_hcu
					dim_mcu = self.dc.dim_mcu
					scale = self.dc.scale
					window = 100
					frame_count = data[len(data)-1][0]+1
					if end>frame_count+1:
						end = frame_count+1
					if end>start:
						H = 20
						W = 20
						mat = np.zeros([H, W, 3], dtype=np.uint8)
						fig, ax = plt.subplots(figsize=[H/2.5, W/2.5])
						
						cax = ax.imshow(mat, interpolation='nearest')
		
						ffmpeg_writer = animation.writers['ffmpeg']
						metadata = dict(title="BCPNN Activity", artist='GSBN')
						writer = ffmpeg_writer(fps=100, metadata=metadata)
						
						self.statusBar().showMessage("Writing to video file, please wait ...")
						with writer.saving(fig, str(filename), 100):
							for cursor in range(start, end):
								print(cursor)
								
								H = 20
								W = 20
								C = 2
								mat = np.zeros([H, W, 3], dtype=np.uint8)
								cnt = np.zeros([H, W, C, 40], dtype=np.uint8)
		
								bh = cursor
								bl = cursor - window + 1
								if bl<0:
									bl=0
		
								for i in range(len(data)):
									d = data[i]
									if d[0]<bl:
										continue
									if d[0]>bh:
										break
			
									for idx in d[1]:
										if(idx>=dim_hcu*dim_mcu or idx<0):
											continue;
										hcu_idx = idx // 40
										h = (hcu_idx//C)//W
										w = (hcu_idx//C)% W
										c = hcu_idx%C
										cnt[h][w][c][idx%40] += 1
		
								for h in range(H):
									for w in range(W):
										c_idx=0
										for c in range(C):
											c_idx = c_idx * 40 + np.argmax(cnt[h][w][C-c-1])
										if c_idx >= len(color_map):
											c_idx = -1
										color = color_map[c_idx]
										for x in range(3):
											mat[h][w][x] = (color >> 8*(2-x)) & 0xff
					
								mat = mat.astype(np.uint8)
								cax.set_data(mat)
								ax.set_title("t="+str(cursor)+" ms")
								writer.grab_frame()
		
						self.statusBar().showMessage("Succesefully saved to video file: "+filename, 2000)
						return
		self.statusBar().showMessage("Quit saving file", 2000)
	
	def fileQuit(self):
		self.close()
	
	def closeEvent(self, ce):
		self.fileQuit()
	
	def about(self):
		QtGui.QMessageBox.about(self, "About", "BCPNN Activity Visualization Tool\n\nThis program belongs to GSBN and is published under the same license of the main program."
		)
	
	def btnRToggled(self):
		if(self.btn_r.isChecked()):
			self.timer.start(10)
			self.btn_b.setEnabled(False)
			self.btn_f.setEnabled(False)
			self.btn_r.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
		else:
			self.timer.stop()
			self.btn_b.setEnabled(True)
			self.btn_f.setEnabled(True)
			self.btn_r.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
	
	def btnBClicked(self):
		self.bar.triggerAction(QtGui.QAbstractSlider.SliderSingleStepSub)
	
	def btnFClicked(self):
		self.bar.triggerAction(QtGui.QAbstractSlider.SliderSingleStepAdd)
	
	def barValueChanged(self):
		self.update_all()
	
	def update_all(self):
		self.label_t.setText(str(self.bar.value())+"/"+str(self.bar.maximum()))
		self.dc.show_pic(self.data, self.bar.value(), 100)


if __name__ == '__main__':
	qApp = QtGui.QApplication(sys.argv)
	aw = ApplicationWindow()
	aw.setWindowTitle("BCPNN Activity Visualization Tool")
	aw.show()
	sys.exit(qApp.exec_())
