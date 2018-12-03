#!/usr/bin/env python
import sys, math  # sys нужен для передачи argv в QApplication
from Particle import Particle, Speed, Coord
from gl_widget import glWidget
from PyQt5 import QtWidgets, QtCore, QtGui, QtOpenGL
import ui_for_particles
import OpenGL.GL as gl
import OpenGL.GLU as glu
import random
import math

particle_system = []
solar_system = []

class ExampleApp(QtWidgets.QMainWindow, ui_for_particles.Ui_MainWindow, QtOpenGL.QGLWidget):
    xRotationChanged = QtCore.pyqtSignal(int)
    yRotationChanged = QtCore.pyqtSignal(int)
    zRotationChanged = QtCore.pyqtSignal(int)

    def __init__(self):
        self.mass = 0
        self.col_ = QtGui.QColor(0,0,0)
        self.pos = QtCore.QPoint()
        self.x_coord = 0
        self.y_coord = 0
        self.z_coord = 0
        self.mouse_pos = 0
        self.xRot = 0
        self.yRot = 0
        self.zoom = 200
        self.object_ = 0
        
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна  
        self.setMouseTracking(True)
        self.gl_widget_ = glWidget(self.openGLWidget, particle_system, solar_system, self)
        #slider
        self.horizontalSlider.setMinimum(1)
        self.horizontalSlider.setMaximum(100)
        
        self.mass = self.horizontalSlider.value()
        self.label_mass.setText(str(self.mass))
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider.valueChanged[int].connect(self.slider_update)
        self.pushButton.clicked.connect(self.generate_particle)
        self.pushButton_2.clicked.connect(self.add_particle)
        self.pushButton_SS.clicked.connect(self.solar_system)
        self.pushButton_stop.clicked.connect(self.del_stop)
        self.setWindowTitle('Задание 1')
        self.lineEdit_N.setText('20')

        self.x_crd.setText('0.1')
        self.y_crd.setText('0.1')
        self.z_crd.setText('0.1')
        self.speed_u.setText('0.5')
        self.speed_v.setText('0.5')
        self.speed_w.setText('0.5')
        self.comboBox.setCurrentIndex(1)
        self.comboBox.currentIndexChanged.connect(self.combobox_updateIndex)
        #цвет частицы
        #self.col_ = QtGui.QColor(34,139,34,1) #цвет пока отключила
        self.pushButton_3.clicked.connect(self.showDialog)
        self.frame.setStyleSheet("QWidget { background-color: %s }"
            % self.col_.name())
        #параметры частицы
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.gl_widget_.update)
        self.timer.start(1000)
        self.show()

    def slider_update(self):
        self.mass = self.horizontalSlider.value()
        self.label_mass.setText(str(self.mass))
    
    def combobox_updateIndex(self):
        self.timer.stop()
        self.comboBox.update()
        print('combobox_getIndex')
        self.timer.start(1000)

    def showDialog(self):
        self.col_ = QtWidgets.QColorDialog.getColor()
        if self.col_.isValid():
            self.frame.setStyleSheet("QWidget { background-color: %s }"
                % self.col_.name())

    def generate_particle(self):
        #self.comboBox.update()
        N_p = int(self.lineEdit_N.text())
        del particle_system [:]
        particle_system.append(Particle(Coord(0,0,0), Speed(0,0,0), 1000, (0.1,0.1,0.8)))
        for i in range(1,N_p):
            crd = Coord(random.randint(-100, 100),random.randint(-100, 100),random.randint(-100, 100))
            vel = Speed(random.randint(-10, 10) / 100000.0, random.randint(-10, 10) / 100000.0, random.randint(-10, 10) / 100000.0)
            mass = random.uniform(1, 500)
            rgb_col = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
            particle_system.append(Particle(crd,vel,mass,rgb_col))
        self.gl_widget_.update()

    def solar_system(self):
        del particle_system [:]
        au = 149597870700
        earth_mass = 5.9726 * 10**24
        m = [] 
        x_ = []
        m.append(332946)
        m.append(0.055)
        m.append(0.815)
        m.append(1.0)
        m.append(0.107)
        m.append(317.8)
        m.append(95.2)
        m.append(14.37)
        m.append(17.15)
        x_.append(0.0)
        x_.append(0.387 * au)
        x_.append(0.723 * au)
        x_.append(1.0 * au)
        x_.append(1.524 * au)
        x_.append(5.204 * au)
        x_.append(9.582 * au)
        x_.append(19.23 * au)
        x_.append(30.10 * au)
        del solar_system [:]
        #Sun r = 695508 | 332946 масс Земли
        sun_ = Particle(Coord(0.0,0.0,0.0), Speed(0.0,0.0,0.0),70.508*5,[1.0, 0.9907072556649119, 0.1983062485694667])
        solar_system.append(sun_)
        particle_system.append(sun_)
        #Mercury r = 2440 dist = 0.3098292 | 0.055 масс Земли
        mercury_ = Particle(Coord(0.387, 0.0, 0.0),Speed(0, 47360, 0),24.40*5,(0.395574883649958, 0.0, 0.005233844510566872))
        solar_system.append(mercury_)
        particle_system.append(mercury_)
        #Venus r = 6052 dist =  0.7195714 | 0.815
        venus_ = Particle(Coord(0.723,0.0,0.00), Speed(0, 35020, 0),60.52*5, (0.9163042648966201, 0.3935606927595941, 0.10646219577325093))
        solar_system.append(venus_)
        particle_system.append(venus_)
        #Earth r = 6371 dist = 0.9859811 | 1
        earth_ = Particle(Coord(1,0.0,0.0000000),Speed(0, 29783, 0),63.71*5,(0.14355687800411993, 0.9709010452429999, 0.20602731364919508))
        solar_system.append(earth_)
        particle_system.append(earth_)
        #Mars r = 3390 dist = 1.4210447 | 0.107
        mars_ = Particle(Coord(1.524,0,0),Speed(0, 24100, 0),33.90*5, (0.9,0.2,0.1))
        solar_system.append(mars_)
        particle_system.append(mars_)
        #Jupiter r = 69911 dist = 5.3593211 | 317.8
        jupiter_ = Particle(Coord(5.204,0,0),Speed(0, 13070, 0),699.911*5,(0.99,0.7,0.05))
        solar_system.append(jupiter_)
        particle_system.append(jupiter_)
        #Saturn r = 58232 dist = 10.0800986 | 95.2
        saturn_ = Particle(Coord(9.582,0,0), Speed(0, 9690, 0), 582.32*5,(0.5,0.6,0.0))
        solar_system.append(saturn_)
        particle_system.append(saturn_)
        #Uranus r = 25362 dist = 19.8681369 | 14.37
        uranus_ = Particle(Coord(19.23,0,0),Speed(0, 6810, 0),253.62*5,(0.1,0.7,0.7))
        solar_system.append(uranus_)
        particle_system.append(uranus_)
        #Neptune r = 24622 dist = 29.9240293 | 17.15
        neptune_ = Particle(Coord(30.10,0,0),Speed(0, 5430, 0),246.22*5,(0,0,0.9))
        solar_system.append(neptune_)
        particle_system.append(neptune_)
        for i in range(len(solar_system)):
            solar_system[i].m = m[i]
        self.gl_widget_.update()

    def add_particle(self):
        global particle_system
        #self.timer.stop()
        coordinates = Coord(float(self.x_crd.text()),float(self.y_crd.text()),float(self.z_crd.text()))
        speed = Speed(float(self.speed_u.text())/100000,float(self.speed_v.text())/100000,float(self.speed_w.text())/10000)
        self.mass = self.horizontalSlider.value() * 7
        p = Particle(coordinates,speed,self.mass,self.col_.getRgbF())
        particle_system.append(p)
        #self.timer.start(1000)
        self.gl_widget_.update()

    def del_stop(self):
        self.timer.stop()
        del particle_system [:]
        del solar_system [:]
        self.gl_widget_.update()
        self.timer.start(1000)

    def animation(self):
        self.timer.timeout.connect(self.gl_widget_.update)
        self.timer.start(1000)

def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    #window.animation()
    app.exec_()  # и запускаем приложение
    
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
