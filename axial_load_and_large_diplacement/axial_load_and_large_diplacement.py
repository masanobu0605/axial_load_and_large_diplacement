import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import time

extensible = 1 #0の場合伸びなし,1の場合伸びあり

##変数定義
r0 = 2
theta_l0 = 20#deg
theta_l0 = theta_l0 * np.pi / 180

##--x0,y0の定義
n = 100 #梁の分割数
x0,y0 = np.zeros(n),np.zeros(n)
theta0 = 0
for i in range(n):
    x0[i] = r0 * np.sin(theta0)
    y0[i] = r0 * (1 - np.cos(theta0))
    theta0 = theta0 + theta_l0/(n-1)
    
l0 = (2 * r0 * np.pi) * (theta_l0 / (2 * np.pi))
d = l0 / 30 #円柱断面形状を想定している．

E = 60*10**9
sigma_y = 210 * 10**6
Zp = d**3/6 #全塑性モーメントの計算に使う．円柱断面形状です
I = np.pi*(d**4)/64
omega = d**2 *np.pi / 4

P_elementcount = 200 #解析要素数
Pload = np.linspace(50,100000,P_elementcount)

result_delta_h = np.zeros(P_elementcount)
result_theta_l = np.zeros(P_elementcount)
lambda_P = np.zeros(P_elementcount) #グラフを作るために用意した，比を表す
lambda_delta = np.zeros(P_elementcount) #グラフを作るために用意した，比を表す

def main(P):
    ##--関数の準備F(start_point,stop_point,)mは(1 + m*sin(theta))になることに注意------------------
    def Finte(a,p):
        n = 500
        a1 = 0
        d_theta = (a - a1) / n
        theta_before = a1
        Fanswer = 0
        
        for i in range(n):
            theta_after = theta_before + d_theta
            Fanswer = Fanswer  + (
                ((1 - (p)*(np.sin(theta_before))**2)**(-0.5)) + 
                ((1 - (p)*(np.sin(theta_after))**2)**(-0.5)) 
                )*d_theta*0.5
            theta_before = theta_after
        return Fanswer
    ##--関数の準備E(start_point,stop_point,m),mは(1 + m*sin(theta))になることに注意 ------------------
    def Einte(a2,p):
        n = 500
        a1= 0
        d_theta = (a2 - a1) / n
        theta_before = a1
        Eanswer = 0
        
        for i in range(n):
            theta_after = theta_before + d_theta
            Eanswer = Eanswer + (
                ((1 - (p)*(np.sin(theta_before))**2)**(0.5)) + 
                ((1 - (p)*(np.sin(theta_after))**2)**(0.5))
                )*d_theta*0.5
            theta_before = theta_after
        return Eanswer
    #--関数の準備第3種楕円積分,kは(1 + k*sin(theta))になることに注意
    def PIinte(n,phi,k):
        nn = 500
        d_theta = (phi - 0) / nn
        theta_before = 0
        PIanswer = 0
        
        for i in range(nn):
            theta_after = theta_before + d_theta
            PIanswer = PIanswer + (
                (1/((1- n* np.sin(theta_before)**2) * np.sqrt(1 - k * np.sin(theta_before)**2))) + 
                (1/((1- n* np.sin(theta_after)**2) * np.sqrt(1 - k * np.sin(theta_after)**2)))
                )*d_theta*0.5
            theta_before = theta_after
        return PIanswer
    ##--関数の準備m
    def m(a):
        return np.sqrt((4 * P * r0**2)/(E * I - 4 * P * r0**2 * (np.sin(a/2))**2))
    
    ##--theta_lの決定ー
    if extensible == 0:
        def theta_l_define_inex(a):
            return (m(a) * np.sqrt(E * I/ P) * Finte(a/2,-(m(a)**2)) - l0)
        theta_l = optimize.bisect(theta_l_define_inex,0,theta_l0)
        # theta_l = optimize.fsolve(theta_l_define_inex,0)
        
    elif extensible == 1:
        n = 2 * P /(E * omega + P)
        def theta_l_define_ex(theta):
            return m(theta) * np.sqrt(E * I/ P) * (1/(1 + P / (E * omega))) * PIinte(n, theta/2, -m(theta)**2) - l0
        theta_l = optimize.bisect(theta_l_define_ex,0,np.pi/2)
        
    ##--delta_hの決定ー #論文中ではdelta_h/2と記載．半分の長さを計算する
    delta_h = m(theta_l) * np.sqrt(E * I/ P) * (
        (1 + 2 * m(theta_l)**(-2)) * Finte(theta_l/2,-m(theta_l)**2) -
        (2 * m(theta_l)**(-2)) * Einte(theta_l/2,-m(theta_l)**2)) -  r0 * np.sin(theta_l0)
    return delta_h,theta_l


for i in range(P_elementcount):
    ##--進捗バー表示
    proguresnum = P_elementcount
    bar = '■'*(i + 1) + "."*(proguresnum- i -1)
    print(f"\r\033[K[\033[33m{bar}\033[39m] {(i + 1)/(proguresnum)*100:.02f}% ({(i + 1)}/{(proguresnum)})", end="")
    
    result_delta_h[i],result_theta_l[i] = main(Pload[i])
    result_theta_l[i] = result_theta_l[i] * 180 / np.pi
    
    if extensible == 0:
        delta_h_max = l0 - r0 * np.sin(theta_l0)
        P_hat = E * I * delta_h_max/(r0**3)
        lambda_delta[i] = result_delta_h[i] / delta_h_max
        lambda_P[i] = Pload[i]/P_hat
        plt_xlim = 1
        plt_savename = "inextensible.png"
    
    elif extensible == 1:
        delta_h_max = l0 - r0 * np.sin(theta_l0)
        P_hat = E * I * delta_h_max/(r0**3)
        lambda_delta[i] = result_delta_h[i] / delta_h_max
        lambda_P[i] = Pload[i]/P_hat
        plt_xlim = lambda_delta[-1]
        plt_savename = "extensible.png"

datasheet = np.ndarray((P_elementcount,5))
datasheet[:,0] = result_theta_l
datasheet[:,1] = result_delta_h
datasheet[:,2] = lambda_delta
datasheet[:,3] = lambda_P
datasheet[:,4] = Pload
datasheet_pd = pd.DataFrame(datasheet,columns=["result_theta_l [deg]","result_delta_h [m]","lambda_delta","lambda_P","Pload"],)
datasheet_pd.to_csv('axial_load_large_diplacement.csv')

def fig():
    
    fig, ax2 = plt.subplots()
    x = lambda_delta
    y1 = lambda_P
    ax2.plot(x, y1,linestyle = 'solid' ,color = 'red')
    ax2.set_xlim(0,plt_xlim)
    ax2.set_title("論文に乗っている図 | L = " + f'{l0:.1f}' + "[m]",fontname = 'MS Gothic')
    ax2.set_xlabel("δh / δmax",fontname = 'MS Gothic')
    ax2.set_ylabel("P / Phat")
    ax2.axvline(x=1)
    plt.savefig(plt_savename)
    
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))#y軸小数点以下3桁表示
    plt.gca().xaxis.get_major_formatter().set_useOffset(False)
    plt.show()
fig()


"""

最終的にはCSVとPNGが出力できるようにする

"""


