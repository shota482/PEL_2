import module.data_flame_m as d_mod
from matplotlib import pyplot as plt
import numpy as np
from numpy import mean
import pandas as pd
import time

PATH_SPACE=1
SENSOR_SPACE=0.02#[mm]
PACH_SPACE=0.2

cut_condition = [800,240,5,1,2]

#no_spray_li=["no_spray",R'C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\jikken\1120LJV\スプレーなし.csv',30,500,0,-6.5,0.175,5,1]
#spray_li   = ["spray",R'C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\jikken\1120LJV\spray.csv',30,500,1,-5.5]
li_0106 =["0106",R'C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\graduatin thesis\jikken\0106\LJV\pach3_dep2\200_kakou\加工面200.csv',30,200,0,-8,SENSOR_SPACE]
contact_li = ["cont",R'C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\graduatin thesis\jikken\接触式\auto$0.csv',50,1000,4,1.0024,0.02,45]

rough_li_XF = ["rough_XF",R'C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\graduatin thesis\jikken\粗さ計\XF.xls']

rough_5=["rough_5",R"C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\graduatin thesis\jikken\粗さ計\5path.xls"]







def ex_excel(df,sheet_name):
    path=R'C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\graduatin thesis\結果\LJV\contact.xlsx'
    df.to_excel(path,sheet_name)


if __name__=="__main__":
    start=time.time()
    data0106=d_mod.ljv(li_0106,cut_condition)
    #data0106.plot()
    data_con=d_mod.contact(contact_li,cut_condition)
    #print(vars(data0106))
    #print(vars(data_con))
    """
    data0106.plot()
    data_con.plot()
    """
    #plt.show()
    
    #r_5 =d_mod.roughness(rough_5,cut_condition)
    #r_XF=d_mod.roughness(rough_li_XF,cut_condition)
    data0106_planar_zone=[[3.75,-8.00],[4.75,7.98]]
    #print(vars(r_XF))
    #\start_pz,stop_pz=data0106_planar_zone
    #data0106.planar_approx(start_pz,stop_pz)
    data0106.planar_approx(data0106_planar_zone)
    #data_title_ljv=[["F_min_v","F_min_indx"],["XF_min_v","XF_min_indx"],"diff"]
    #min_info_0106=data0106.find_min_point([20,-2.5],[36,2.5])
    #data0106.ex_excel(min_info_0106)
    #ex_excel(min_info_0106[2],"LJV_minvalue","sheet")
    data_con.path_count=4
    min_info_contact = data_con.find_min_point([10,0],[40,-4])
    data_con.ex_excel(min_info_contact)
    #ex_excel(min_info_contact[0][0],"LJV_minvalue","sheet")
    #r_XF.rough_data_pro(XF=True)
    #r_5.df.plot()
    #r_5.rough_data_pro()

    end=time.time()
    
    process_time=end-start
    print("process time:",process_time)

    plt.show()



