from Particle import Particle, Speed, Coord
import ui_for_particles
import numpy as np
import math
import time
import scipy.integrate
import threading
from multiprocessing import Barrier
import cythv
import pyopencl as cl
import os
os.environ["PYOPENCL_CTX"]="0:0" #CPU

class verlet_solvers(ui_for_particles.Ui_MainWindow):
    def __init__(self, list_p, gui):
        self.list_of_particles = list_p
        self.gui = gui
        self.T = 25
        self.M = 5

    def verlet__(self):
        start_time = time.clock()
        G = 6.67408e-11
        module = 0
        Time_ = 600000
        delta_t = 6000
        time_iter = np.linspace(0,Time_,delta_t)
        flag = 0
        n = len(self.list_of_particles)
        if n<2:
            return
        print(len(time_iter))
        t = 0
        #print('T ================== ',t)
        x_ = []
        y_ = []
        z_ = []
        x_for_opengl = []
        y_for_opengl = []
        z_for_opengl = []
        vx_n = []
        vy_n = []
        vz_n = []
        m = []
        m_for_opengl = []
        color_ = []

        ax_n = []
        ay_n = []
        az_n = []
        ax_n1 = []
        ay_n1 = []
        az_n1 = []

        #print('кол-во частиц', n)
        for t in range (1,self.M):
            del x_ [:] 
            del y_ [:]
            del z_ [:]
            del x_for_opengl [:]
            del y_for_opengl [:]
            del z_for_opengl [:]
            del vx_n [:]
            del vy_n [:]
            del vz_n [:]
            del m [:]
            del m_for_opengl [:]
            del color_ [:]

            del ax_n [:]
            del ay_n [:]
            del az_n [:]
            del ax_n1 [:]
            del ay_n1 [:]
            del az_n1 [:] 
            for p1 in self.list_of_particles:
                for p2 in self.list_of_particles:
                    if (math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2) > 0) & (math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2) < (p1.m + p2.m)/100):
                        if p1.m > p2.m:
                            p1.m += p2.m
                        else:
                            p1.is_active = False

            for p in self.list_of_particles:
                if p.is_active == True:
                    x_.append(p.x)
                    y_.append(p.y)
                    z_.append(p.z)
                    vx_n.append(p.xv)
                    vy_n.append(p.yv)
                    vz_n.append(p.zv)
                    m.append(p.m)
                    color_.append(p.color)  

            n_ = len(x_)
            #print('кол-во частиц = ', n_)
            k = 0
            j = 0
            for j in range(n_):
                ax_ = 0
                ay_ = 0
                az_ = 0
                for k in range(n_):
                    if k != j:
                        #print('модуль', math.sqrt((x_[k]-x_[j])**2 + (y_[k]-y_[j])**2 + (z_[k]-z_[j])**2)**3, ' ',k,' ', j)
                        ax_ += m[k] * G * (x_[k] - x_[j]) / math.sqrt((x_[k]-x_[j])**2 + (y_[k]-y_[j])**2 + (z_[k]-z_[j])**2)**3
                        ay_ += m[k] * G * (y_[k] - y_[j]) / math.sqrt((x_[k]-x_[j])**2 + (y_[k]-y_[j])**2 + (z_[k]-z_[j])**2)**3
                        az_ += m[k] * G * (z_[k] - z_[j]) / math.sqrt((x_[k]-x_[j])**2 + (y_[k]-y_[j])**2 + (z_[k]-z_[j])**2)**3
                ax_n.append(ax_)
                ay_n.append(ay_)
                az_n.append(az_)
        
            x_n1 = [x_n + u_n*delta_t + 0.5*a_n*delta_t**2 for x_n,u_n,a_n in zip(x_,vx_n,ax_n)]
            y_n1 = [y_n + v_n*delta_t + 0.5*a_n*delta_t**2 for y_n,v_n,a_n in zip(y_,vy_n,ay_n)]
            z_n1 = [z_n + w_n*delta_t + 0.5*a_n*delta_t**2 for z_n,w_n,a_n in zip(z_,vz_n,az_n)]

            k = 0
            j = 0
            for j in range(n_):
                ax_ = 0
                ay_ = 0
                az_ = 0
                for k in range(n_):
                    if k != j:
                        ax_ += G*m[k]* (x_n1[k]-x_n1[j]) / math.sqrt((x_n1[k]-x_n1[j])**2 + (y_n1[k]-y_n1[j])**2 + (z_n1[k]-z_n1[j])**2)**3
                        ay_ += G*m[k]* (y_n1[k]-y_n1[j]) / math.sqrt((x_n1[k]-x_n1[j])**2 + (y_n1[k]-y_n1[j])**2 + (z_n1[k]-z_n1[j])**2)**3
                        az_ += G*m[k]* (z_n1[k]-z_n1[j]) / math.sqrt((x_n1[k]-x_n1[j])**2 + (y_n1[k]-y_n1[j])**2 + (z_n1[k]-z_n1[j])**2)**3
                ax_n1.append(ax_)
                ay_n1.append(ay_)
                az_n1.append(az_)
        
            vx_n1 = [u + 0.5*(an + an1)*delta_t for u, an, an1 in zip(vx_n, ax_n, ax_n1)]
            vy_n1 = [v + 0.5*(an + an1)*delta_t for v, an, an1 in zip(vy_n, ay_n, ay_n1)]
            vz_n1 = [w + 0.5*(an + an1)*delta_t for w, an, an1 in zip(vz_n, az_n, az_n1)]

            del self.list_of_particles [:]
            for i in range(n_):
                pos_ = Coord(x_n1[i],y_n1[i],z_n1[i])
                speed_ = Speed(vx_n1[i],vy_n1[i],vz_n1[i])
                self.list_of_particles.append(Particle(pos_,speed_,m[i],color_[i]))
        computing_time = time.clock()-start_time
        print(time.clock()-start_time)
        self.gui.label_time.setText(str(computing_time))
    def for__scipy(self,init,t,n,mass):
        print('зашел в for_scipy')
        G = 6.67408e-11
        x_0 = []
        y_0 = []
        z_0 = []
        vx_0 = []
        vy_0 = []
        vz_0 = []
        #m = []
        color_ = []

        ax_n = []
        ay_n = []
        az_n = []
        ax_n1 = []
        ay_n1 = []
        az_n1 = []

        del x_0 [:] 
        del y_0 [:]
        del z_0 [:]
        del vx_0 [:]
        del vy_0 [:]
        del vz_0 [:]
        #del m [:]
        del color_ [:]

        del ax_n [:]
        del ay_n [:]
        del az_n [:]
        del ax_n1 [:]
        del ay_n1 [:]
        del az_n1 [:] 

        module = 0
        for i in range(n):
            x_0.append(init[i*3])
            y_0.append(init[i*3 + 1])
            z_0.append(init[i*3 + 2])
            vx_0.append(init[i*3 + 3*n])
            vy_0.append(init[i*3 + 3*n + 1])
            vz_0.append(init[i*3 + 3*n + 2])
        result = []
        del result [:]
        n_ = n
        for i in range(n_):
            result.append(vx_0[i])
            result.append(vy_0[i])
            result.append(vz_0[i])
        for j in range(n_):
            ax_ = 0
            ay_ = 0
            az_ = 0
            for k in range(n_):
                if k != j:
                    ax_ += mass[k] * G * (x_0[k] - x_0[j]) / math.sqrt((x_0[k]-x_0[j])**2 + (y_0[k]-y_0[j])**2 + (z_0[k]-z_0[j])**2)**3
                    ay_ += mass[k] * G * (y_0[k] - y_0[j]) / math.sqrt((x_0[k]-x_0[j])**2 + (y_0[k]-y_0[j])**2 + (z_0[k]-z_0[j])**2)**3
                    az_ += mass[k] * G * (z_0[k] - z_0[j]) / math.sqrt((x_0[k]-x_0[j])**2 + (y_0[k]-y_0[j])**2 + (z_0[k]-z_0[j])**2)**3
            result.append(ax_)
            result.append(ay_)
            result.append(az_)
        return result
    def verlet__scipy(self):
        start_time = time.clock()
        T = 6000*4 #Расчетное время - шаг на самом деле это шаааааааг
        M = 5 #сколько точек 
        time_span = np.linspace(0, T, M)
        mass = []
        xyz = []
        Vxyz = []
        color_ = []
        res = []

        del mass [:]
        del xyz [:]
        del Vxyz [:]
        del color_ [:]
        del res [:]
        
        n = len(self.list_of_particles)
        #print('кол-во частиц', n)
        if n<2:
            return
        for p1 in self.list_of_particles:
            for p2 in self.list_of_particles:
                if (math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2) > 0) & (math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2) < (p1.m + p2.m)/100):
                    if p1.m > p2.m:
                        p1.m += p2.m
                    else:
                        p1.is_active = False

        for p in self.list_of_particles:
            if p.is_active == True:
                xyz.append(p.x)
                xyz.append(p.y)
                xyz.append(p.z)
                Vxyz.append(p.xv)
                Vxyz.append(p.yv)
                Vxyz.append(p.zv)
                mass.append(p.m)
                color_.append(p.color)  

        n_ = len(color_)
        init_cond = sum([list(map(float, xyz)), list(map(float, Vxyz))], [])
        del res [:]
        res = scipy.integrate.odeint(self.for__scipy, init_cond, time_span, args=(n_,list(map(float,mass))))
        
        x_n1 =[]
        y_n1 =[]
        z_n1 = []
        vx_n1 =[]
        vy_n1 = []
        vz_n1 = []
        del x_n1 [:]
        del y_n1 [:]
        del z_n1 [:]
        del vx_n1 [:]
        del vy_n1 [:]
        del vz_n1 [:]
        for i in range(1,len(res)):
            for j in range(n_):
                x_n1.append(res[i][j*3])
                y_n1.append(res[i][j*3+1])
                z_n1.append(res[i][j*3+2])
                vx_n1.append(res[i][j*3 + 3*n_])
                vy_n1.append(res[i][j*3 + 3*n_ + 1])
                vz_n1.append(res[i][j*3 + 3*n_ + 2])

        del self.list_of_particles [:]
        for i in range(n_):
            pos_ = Coord(x_n1[i],y_n1[i],z_n1[i])
            speed_ = Speed(vx_n1[i],vy_n1[i],vz_n1[i])
            self.list_of_particles.append(Particle(pos_,speed_,mass[i],color_[i]))
        computing_time = time.clock()-start_time
        print(time.clock()-start_time)
        self.gui.label_time.setText(str(computing_time))
    class MyThread(threading.Thread):
        def __init__(self, bar1,bar2,n, M, j, dt, G, x, vx, m, axm, y, vy, aym, z, vz, azm):
            threading.Thread.__init__(self)
            self.bar1 = bar1
            self.bar2 = bar2
            self.n = n
            self.M_ = M
            self.j = j
            self.dt = dt
            self.G = G
            self.vx = vx
            self.x = x
            self.m = m
            self.axm = axm
            self.y = y
            self.vy = vy
            self.aym = aym
            self.z = z
            self.vz = vz
            self.azm = azm
        
        def run(self):
            for i in range(1,self.M_):
                msg = "%s is running" % self.name
                #print(msg)
                delta_t = 6000
                self.x[i * self.n + self.j] = self.x[(i - 1) * self.n + self.j] + \
                              self.vx[(i - 1) * self.n + self.j] * self.dt + 1.0 / 2 * self.axm[
                                  (i - 1) * self.n + self.j] * self.dt ** 2
                self.y[i * self.n + self.j] = self.y[(i - 1) * self.n + self.j] + \
                              self.vy[(i - 1) * self.n + self.j] * self.dt + 1.0 / 2 * self.aym[
                                  (i - 1) * self.n + self.j] * self.dt ** 2
                self.z[i * self.n + self.j] = self.z[(i - 1) * self.n + self.j] + \
                              self.vz[(i - 1) * self.n + self.j] * self.dt + 1.0 / 2 * self.azm[
                                  (i - 1) * self.n + self.j] * self.dt ** 2
                ax = 0
                ay = 0
                az = 0
                self.bar1.wait()
                for f in range(self.n):
                    if f != self.j:
                        ax += self.m[f] * self.G * (self.x[i * self.n + f] - self.x[i * self.n + self.j]) / \
                              math.sqrt((self.x[i * self.n + f] - self.x[i * self.n + self.j])**2 + (self.y[i * self.n + f] - self.y[i * self.n + self.j])**2 + (self.z[i * self.n + f] - self.z[i * self.n + self.j])**2) ** 3
                        ay += self.m[f] * self.G * (self.y[i * self.n + f] - self.y[i * self.n + self.j]) /\
                              math.sqrt((self.x[i * self.n + f] - self.x[i * self.n + self.j])**2 + (self.y[i * self.n + f] - self.y[i * self.n + self.j])**2 + (self.z[i * self.n + f] - self.z[i * self.n + self.j])**2) ** 3
                        az += self.m[f] * self.G * (self.z[i * self.n + f] - self.z[i * self.n + self.j]) /\
                              math.sqrt((self.x[i * self.n + f] - self.x[i * self.n + self.j])**2 + (self.y[i * self.n + f] - self.y[i * self.n + self.j])**2 + (self.z[i * self.n + f] - self.z[i * self.n + self.j])**2) ** 3
                self.axm[i * self.n + self.j] = ax
                self.aym[i * self.n + self.j] = ay
                self.azm[i * self.n + self.j] = az
                self.vx[i * self.n + self.j] = self.vx[(i - 1) * self.n + self.j] + 1.0 / 2 * self.dt * (self.axm[i * self.n + self.j] + self.axm[(i - 1) * self.n + self.j])
                self.vy[i * self.n + self.j] = self.vy[(i - 1) * self.n + self.j] + 1.0 / 2 * self.dt * (self.aym[i * self.n + self.j] + self.aym[(i - 1) * self.n + self.j])
                self.vz[i * self.n + self.j] = self.vz[(i - 1) * self.n + self.j] + 1.0 / 2 * self.dt * (self.azm[i * self.n + self.j] + self.azm[(i - 1) * self.n + self.j])
            self.bar2.wait()
    def verlet_threading(self):
        start_time = time.clock()
        print('VERLET_THREADING')
        mass = []
        xyz = []
        Vxyz = []
        color_ = []
        res = []
        del mass [:]
        del xyz [:]
        del Vxyz [:]
        del color_ [:]
        del res [:]
        n_ = len(self.list_of_particles)
        print('тут', n_)
        bar1 = threading.Barrier(n_)
        G = 6.67408e-11
        T = 6000 #Расчетное время
        dt = 6000 
        M = 5 #сколько точек 
        time_span = np.linspace(0, T, M)
        x = np.zeros(n_*M)
        y = np.zeros(n_*M)
        z = np.zeros(n_*M)
        vx = np.zeros(n_*M)
        vy = np.zeros(n_*M)
        vz = np.zeros(n_*M)
        i = 0
        for p in self.list_of_particles:
            x[i] = p.x
            y[i] = p.y
            z[i] = p.z
            vx[i] = p.xv
            vy[i] = p.yv
            vz[i] = p.zv
            mass.append(p.m)
            color_.append(p.color)
            i = i+1
        axm = np.zeros(n_*M)
        aym = np.zeros(n_*M)
        azm = np.zeros(n_*M)
        for j in range(n_):
            ax_ = 0
            ay_ = 0
            az_ = 0
            for k in range(n_):
                if k != j:
                    #print('модуль', math.sqrt((x_[k]-x_[j])**2 + (y_[k]-y_[j])**2 + (z_[k]-z_[j])**2)**3)
                    ax_ += mass[k] * G * (x[k] - x[j]) / math.sqrt((x[k]-x[j])**2 + (y[k]-y[j])**2 + (z[k]-z[j])**2)**3
                    ay_ += mass[k] * G * (y[k] - y[j]) / math.sqrt((x[k]-x[j])**2 + (y[k]-y[j])**2 + (z[k]-z[j])**2)**3
                    az_ += mass[k] * G * (z[k] - z[j]) / math.sqrt((x[k]-x[j])**2 + (y[k]-y[j])**2 + (z[k]-z[j])**2)**3
            axm[j] = ax_
            aym[j] = ay_
            azm[j] = az_
        bar2 = threading.Barrier(n_+1)
        for j in range(n_):
            thread = self.MyThread(bar1,bar2, n_, M, j, dt, G, x, vx, mass, axm, y, vy, aym, z, vz, azm)
            thread.start()
        bar1.wait()
        computing_time = time.clock()-start_time
        print(time.clock()-start_time)
        self.gui.label_time.setText(str(computing_time))
    
    def verlet_cython(self):
        print('cython')
        G = 6.67408e-11
        T = 6000 #Расчетное время
        dt = 6000 
        M = 5 #сколько точек 
        n_ = len(self.list_of_particles)
        mass = []
        xyz = []
        Vxyz = []
        color_ = []
        res = []
        del mass [:]
        del xyz [:]
        del Vxyz [:]
        del color_ [:]
        del res [:]
        x = np.zeros(n_*M)
        y = np.zeros(n_*M)
        z = np.zeros(n_*M)
        vx = np.zeros(n_*M)
        vy = np.zeros(n_*M)
        vz = np.zeros(n_*M)
        axm = np.zeros(n_*M)
        aym = np.zeros(n_*M)
        azm = np.zeros(n_*M)
        i = 0
        for p in self.list_of_particles:
            x[i] = p.x
            y[i] = p.y
            z[i] = p.z
            vx[i] = p.xv
            vy[i] = p.yv
            vz[i] = p.zv
            mass.append(p.m)
            color_.append(p.color)
            i = i+1
        m1 = np.array(mass)
        import pyximport
        pyximport.install(setup_args={'include_dirs': np.get_include()})
        import cythv
        start_time = time.clock()
        result = cythv.cython_solver(n_, G, dt, m1, M, x, y, z, vx, vy, vz, axm, aym, azm)
        computing_time = time.clock() - start_time
        print(time.clock()-start_time)
        self.gui.label_time.setText(str(computing_time))
    
    def verlet_opencl(self):
        start_time = time.clock()
        print('opencl')
        G = 6.67408e-11
        T = 6000 #Расчетное время
        dt = 6000 
        M = 5 #сколько точек 
        n_ = len(self.list_of_particles)
        n = n_
        mass = []
        xyz = []
        Vxyz = []
        color_ = []
        res = []
        del mass [:]
        del xyz [:]
        del Vxyz [:]
        del color_ [:]
        del res [:]
        x = np.zeros(n_*M)
        y = np.zeros(n_*M)
        z = np.zeros(n_*M)
        vx = np.zeros(n_*M)
        vy = np.zeros(n_*M)
        vz = np.zeros(n_*M)
        axm = np.zeros(n_*M)
        aym = np.zeros(n_*M)
        azm = np.zeros(n_*M)
        i = 0
        for p in self.list_of_particles:
            x[i] = p.x
            y[i] = p.y
            z[i] = p.z
            vx[i] = p.xv
            vy[i] = p.yv
            vz[i] = p.zv
            mass.append(p.m)
            color_.append(p.color)
            i = i+1
        m1 = np.array(mass)
        for j in range(n_):
            ax = 0
            ay = 0
            az = 0
            for f in range(n_):
                if f != j:
                    ax += mass[f] * G * (x[f] - x[j]) / math.sqrt((x[f] - x[j])**2 + (y[f] - y[j])**2 + (z[f] - z[j])**2) ** 3
                    ay += mass[f] * G * (y[f] - y[j]) / math.sqrt((x[f] - x[j])**2 + (y[f] - y[j])**2 + (z[f] - z[j])**2) ** 3
                    az += mass[f] * G * (y[f] - y[j]) / math.sqrt((x[f] - x[j])**2 + (y[f] - y[j])**2 + (z[f] - z[j])**2) ** 3
            axm[j] = ax
            aym[j] = ay
            azm[j] = az
        source = """
        kernel void compute(int n, int tn, double G, double dt, __global double *m, __global double *x, __global double *y, __global double *z, __global double *vx
        , __global double *vy, __global double *vz, __global double *axm, __global double *aym, __global double *azm){
            double ax = 0;
            double ay = 0;
            double az = 0;
            int j = get_global_id(0);
            for(int i = 1; i<tn; i++) {
                x[i * n + j] = x[(i - 1) * n + j] + vx[(i - 1) * n + j] * dt + 1.0 / 2 * axm[
                                (i - 1) * n + j] * pow(dt, 2);
                y[i * n + j] = y[(i - 1) * n + j] + vy[(i - 1) * n + j] * dt + 1.0 / 2 * aym[
                                (i - 1) * n + j] * pow(dt, 2);
                z[i * n + j] = z[(i - 1) * n + j] + vz[(i - 1) * n + j] * dt + 1.0 / 2 * azm[
                                (i - 1) * n + j] * pow(dt, 2);
                barrier(CLK_GLOBAL_MEM_FENCE);
                ax = 0;
                ay = 0;
                az = 0;
                for(int f = 0; f<n; f++) {
                    if(f != j) {
                        ax += m[f] * G * (x[i * n + f] - x[i * n + j]) /
                              pow(sqrt(pow(x[i * n + f] - x[i * n + j], 2) + pow(y[i * n + f] - y[i * n + j], 2) + pow(z[i * n + f] - z[i * n + j], 2)), 3);
                        ay += m[f] * G * (y[i * n + f] - y[i * n + j]) /
                              pow(sqrt(pow(x[i * n + f] - x[i * n + j], 2) + pow(y[i * n + f] - y[i * n + j], 2) + pow(z[i * n + f] - z[i * n + j], 2)), 3);
                        az += m[f] * G * (z[i * n + f] - z[i * n + j]) /
                              pow(sqrt(pow(x[i * n + f] - x[i * n + j], 2) + pow(y[i * n + f] - y[i * n + j], 2) + pow(z[i * n + f] - z[i * n + j], 2)), 3);
                    }       
                }
                axm[i * n + j] = ax;
                aym[i * n + j] = ay;
                azm[i * n + j] = az;
                vx[i * n + j] = vx[(i - 1) * n + j] + 1.0 / 2 * dt * (axm[i * n + j] + axm[(i - 1) * n + j]);
                vy[i * n + j] = vy[(i - 1) * n + j] + 1.0 / 2 * dt * (aym[i * n + j] + aym[(i - 1) * n + j]);
                vz[i * n + j] = vz[(i - 1) * n + j] + 1.0 / 2 * dt * (azm[i * n + j] + azm[(i - 1) * n + j]);
            }
        }"""
        #Initialization phase:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        #Create mem object:
        xbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        ybuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y)
        zbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=z)
        vxbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=vx)
        vybuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=vy)
        vzbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=vz)
        axmbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=axm)
        aymbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=aym)
        azmbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=azm)
        mbuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=m1)
        lengtht = np.int32(M)
        lengthn = np.int32(n_)
        Gb = np.float(G)
        dtb = np.float(dt)
        #Compilation phase
        prg = cl.Program(ctx, source)
        try:
            prg.build()
        except:
            print("Error:")
            print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
            raise
        #Execution phase:
        start_time = time.time()
        prg.compute(queue, (1,), None, lengthn, lengtht, Gb, dtb, mbuf, xbuf, ybuf, zbuf, vxbuf, vybuf, vzbuf, axmbuf, aymbuf, azmbuf).wait()
        computing_time = time.clock() - start_time
        print(time.clock()-start_time)
        self.gui.label_time.setText(str(computing_time))
        resultx = np.empty_like(x)
        resulty = np.empty_like(y)
        resultz = np.empty_like(z)
        resultvx = np.empty_like(vx)
        resultvy = np.empty_like(vy)
        resultvz = np.empty_like(vz)
        resultaxm = np.empty_like(axm)
        resultaym = np.empty_like(aym)
        resultazm = np.empty_like(azm)
        cl.enqueue_copy(queue, resultx, xbuf).wait()
        cl.enqueue_copy(queue, resulty, ybuf).wait()
        cl.enqueue_copy(queue, resulty, zbuf)
        cl.enqueue_copy(queue, resultvx, vxbuf)
        cl.enqueue_copy(queue, resultvy, vybuf)
        cl.enqueue_copy(queue, resultvy, vzbuf)
        cl.enqueue_copy(queue, resultaxm, axmbuf)
        cl.enqueue_copy(queue, resultaym, aymbuf)
        cl.enqueue_copy(queue, resultaym, azmbuf)
        # self.countError(nStr, mStr, xyStr, vStr, resultx, resulty, n)
