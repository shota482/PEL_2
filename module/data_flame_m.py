from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D 
from . import const
from scipy.signal import find_peaks
import openpyxl
import os




class data():
    def __init__(self,data_li,condi_li):
        self.name          = data_li[const.FILE_NAME]
        self.file_adrs     = data_li[const.FILE_ADRS]
        self.feed          = data_li[const.FEED]
        self.sampling_rate = data_li[const.SAMPLING_RATE]
        self.hight_surface = data_li[const.DEP_S]
        self.sample_space_xf = data_li[const.SAMPLE_SPACE_XF]
        self.pach_space    = condi_li[const.CUT_FEED]/condi_li[const.SPINDLE]
        self.path_count    = condi_li[const.PATH_COUNT]
        self.path_space    = condi_li[const.PACH_XF]
        self.depth_cut     = condi_li[const.DEPTH_CUT]  #temp
        self.sampling_distance = (self.feed/60)/self.sampling_rate
        self.df=pd.DataFrame()

    def plot(self):
        self.fig = plot_3d(self.df,self.name)
        
    def df_cut(self,pos_start,pos_stop):
        start_indx,start_col    = pos_start
        stop_indx,stop_col      = pos_stop
        #print(self.df.loc[start_indx:stop_indx,start_col:stop_col])
        return self.df.loc[start_indx:stop_indx,start_col:stop_col]
    
    def find_min_point(self,pos_start,pos_stop):
        target_df=self.df_cut(pos_start,pos_stop).copy().T
        plot_3d(target_df,"")
        plt.show()
        XF_min_V_df,XF_min_IDX_df,XF_csp_df,XF_csp_IDX_df=find_peak_XF(target_df,self.path_space,self.sample_space_xf,self.path_count)
        index_marker(target_df,XF_min_IDX_df,  target_df.columns[0] )
        target_peak_FD= XF_min_V_df.T
        #print(target_peak_FD)
        FD_min_V_df,FD_min_IDX_df,FD_csp_df,FD_csp_IDX_df=find_peak_FD(target_peak_FD,self.pach_space,self.sampling_distance,pos_stop[0]-pos_start[0])
        index_marker(target_peak_FD,FD_min_IDX_df,  0 )
        diff_df = FD_min_V_df.copy()
        
        for column in diff_df.columns:
            diff_df[column] = 2-(self.hight_surface-FD_min_V_df[column])
        diff_df=diff_df.applymap(lambda x: np.nan if x==2-self.hight_surface  else x)
        #print(diff_df)
        print(diff_df.mean())
        return[[FD_min_V_df,XF_min_V_df],[FD_min_IDX_df,XF_min_IDX_df],[FD_csp_df,XF_csp_df],[FD_csp_IDX_df,XF_csp_IDX_df],diff_df]
    
    def ex_excel(self,data):
        min_v,min_idx,csp_v,csp_idx,diff=data
        FD_idx,XF_idx=min_idx
        FD_v,XF_v = min_v
        FD_csp_idx,XF_csp_idx=csp_idx
        FD_csp,XF_csp=csp_v
        path = os.path.join(R"C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\graduatin thesis\結果\LJV",self.name+".xlsx")
        with pd.ExcelWriter(path) as writer:
            FD_idx.to_excel(writer,sheet_name="FD_index")
            FD_v.to_excel(writer,sheet_name="FD_min")
            XF_idx.T.to_excel(writer,sheet_name="XF_index")
            XF_v.T.to_excel(writer,sheet_name="XF_min")
            XF_csp_idx.T.to_excel(writer,sheet_name="XF_csp_index")
            XF_csp.T.to_excel(writer,sheet_name="XF_csp")
            FD_csp_idx.to_excel(writer,sheet_name="FD_csp_index")
            FD_csp.to_excel(writer,sheet_name="FD_csp")

            diff.to_excel(writer,sheet_name="diff")
        
    

class ljv(data):
    def __init__(self,data_li,condi_li):
        super().__init__(data_li,condi_li)
        self.depth_valley  = data_li[const.DEP_V]
        self.df            = make_df_ljv(self.file_adrs,self.sampling_distance,self.hight_surface,self.depth_valley)

    def planar_approx(self,zone):
        pos_start,pos_stop = zone
        start_indx,start_col    = pos_start
        stop_indx,stop_col      = pos_stop
        palane_df=self.df.loc[start_indx:stop_indx,start_col:stop_col].copy()
        palane_df = palane_df.dropna(axis=1,how='all')
        palane_df = palane_df.dropna(how='any')
        count_nan = palane_df.isnull().values.sum()

        if count_nan:
            print("include null data in planar dataframe nun count:",count_nan)
            return 0
        else:
            a,b,_ = approx(palane_df)
            print("a=",a,"b=",b)

            self.df = fix_deg(self.df,a,b)
            self.hight_surface = palane_df.mean(axis=1).mean()

            std_XF = palane_df.std(axis=1).mean()
            std_F  = palane_df.std().mean()
            print("surface_hight =",self.hight_surface)
            print("std_XF:",std_XF,"  std_F:",std_F)
            return 0 #a,b,c [z=ax+by+c]
        



class contact(data):
    def __init__(self,data_li,condi_li):
        super().__init__(data_li,condi_li)
        self.calib_const   = data_li[const.CALIB_CONST]
        self.sample_len    =data_li[const.SAMPLE_LEN]

        self.df            = make_df_contact(self.file_adrs,self.calib_const,self.sampling_distance,self.sample_len)

class roughness(data):
    def __init__(self,data_li,condi_li):#propatyは継承しない
        self.name          = data_li[const.FILE_NAME]
        self.file_adrs     = data_li[const.FILE_ADRS]
        self.pach_space    = condi_li[const.CUT_FEED]/condi_li[const.SPINDLE]
        self.path_count    = condi_li[const.PATH_COUNT]
        self.path_space    = condi_li[const.PACH_XF]
        self.df            = make_df_rough(self.file_adrs)
        self.sampling_distance = 0.0005
        
    def rough_data_pro(self,XF=False):
        if XF:
            self.df,min_info,csp_info=re_size_rough(self.df,self.path_space,self.sampling_distance)
        else:
            self.df,min_info,csp_info=re_size_rough(self.df,self.pach_space,self.sampling_distance)
        print(min_info)
        print(csp_info)
        path = os.path.join(R"C:\Users\shota05\OneDrive - Kyushu Institute Of Technolgy\PEL\graduatin thesis\結果\粗さ",self.name+".xlsx")
        with pd.ExcelWriter(path) as writer:
            self.df.to_excel(writer,sheet_name="data")
            min_info.to_excel(writer,sheet_name="min_info")
            csp_info.to_excel(writer,sheet_name="csp_info")
            



    
##LJV_def###############################
def make_df_ljv(file_adrs,sampling_distance,surface,valley):
    df = csv_to_df(file_adrs,sampling_distance)
    df = eliminate_error_ljv(df,surface,valley)
    return df

def csv_to_df(file_adrs,sampling_distance):
    drop_index_ls=[]
    drop_index_ls.append("蓄積日時")
    for i in range(1,17):
        drop_index_ls.append("OUT"+str(i))

    df=pd.read_csv(file_adrs,encoding='shift-jis')
    df = df.drop(drop_index_ls,axis=1)
    df.columns = df.columns.astype(float)
    df.index=df.index * sampling_distance
    return df

def eliminate_error_ljv(df,surface,valley):
    df = df.mask((df <= -98),np.nan)
    
    df = df.mask((df>surface+0.5),np.nan)
    df = df.mask((df<valley-0.5) ,np.nan)
    el_erro_df = df.dropna(how='all')
    #min_idx=df.idxmin().mean()
    return el_erro_df

def fix_deg(df,a,b):
    for column in df.columns:
        df[column] = -column*a+df[column] #クロスフィード方向の角度修正
    df = df.T                             #転置
    cos_a = np.sqrt(1/((a**2)+1))
    df.index=df.index/cos_a

    for column in df.columns:            
        df[column] = -column*b+df[column] #フィード方向の角度修正
    df = df.T                             #転置
    cos_b = np.sqrt(1/((b**2)+1))
    df.index=df.index/cos_b
    return df   #参照であればインスタンスが書き換わるため不要

def approx(df):
    x_v, y_v = np.meshgrid(df.columns,df.index)
    x_v = x_v.flatten()  # x の値を1次元化
    y_v = y_v.flatten()  # y の値を1次元化
    z_v = df.values.flatten()  # z の値を1次元化
    A = np.c_[x_v, y_v, np.ones_like(x_v)]  # x, y, 定数項を追加
    coeff, _, _, _ = np.linalg.lstsq(A, z_v, rcond=None)
    return coeff
##LJV_def_end#################

##contact_def##################
def make_df_contact(file_adrs,calib_const,sampling_distance,sample_len):
    one_cycle_data_count=sample_len/sampling_distance
    col_name=range(1,7,1)
    df=pd.read_csv(file_adrs,names=col_name,encoding='shift-jis',skiprows=64,skipfooter=3,engine="python")
    #print(df.index,"print")
    df = df.iloc[65:-3,[2]]#65行目~-3行まで　2列
    df = df.reset_index(drop=True)
    df = df.astype(float)*calib_const*3.0077
    df = split_xf_dir(df,5,4,one_cycle_data_count)
    df = eliminate_error_cont(df,5.0)
    df.index=df.index * sampling_distance
    plot_3d(df,"")
    plt.show()
    return df

def split_xf_dir(df,thresh_high,thresh_low,one_cycle_data_count):
    split_df=pd.DataFrame()
    if df[df[3]<thresh_low].index[-1] < df[df[3]<thresh_high].index[-1]:
        df=df.loc[:df[df[3]<thresh_low].index[-1]]
    column = 0
    while len(df[df[3]>5.05].index):
        split_end_indx=int(df[df[3]>thresh_high].index[-1])
        split_start_indx = int(split_end_indx-one_cycle_data_count)
        split_df[column]=df[split_start_indx:split_end_indx][3].copy().tolist()
        column=column-0.02
        df=df.loc[:split_start_indx]
    print(split_df)
    return split_df

def eliminate_error_cont(df,low):
    df = df.mask((df <= low),np.nan)
    df=df.dropna(how='all')
    df=df.dropna(axis=1,how='all')

    return df

##contact_def_end####################

##roughness############################

def make_df_rough(file_adrs):
    df = pd.read_excel(file_adrs, sheet_name="DATA",header=None, index_col=4,engine='xlrd')
    df=df.iloc[:,4]*-1
    return df.to_frame()

def find_path_rough(df,peak_space,sampling_dist):
        dist = peak_space/sampling_dist*0.7
        peak_li,_=find_peaks(df[5],distance=dist)
        peak_index=[]
        peak_name=[]
        min_v=[]
        min_idx=[]
        print(dist)
        index_bf=0
        first_Flag=True
        for index in peak_li:
            index_diff=index-index_bf
            if index_diff>dist+100:
                if first_Flag:
                    peak_index.append(index_bf)
                    peak_name.append(df.iloc[index_bf].name) 
                    first_Flag=False
                peak_index.append(index)
                peak_name.append(df.iloc[index].name)
            index_bf=index

        for pach_num in range(len(peak_index)-1):
            min_v.append(find_min(df,5,peak_index[pach_num],peak_index[pach_num+1]))
            min_idx.append(find_min_idx(df,5,peak_index[pach_num],peak_index[pach_num+1]))
        print(min_v,min_idx)
               #value,name, ,index     ,name
        return min_v,min_idx,peak_name

def re_size_rough(df,peak_space,sampling_dist):
        min_v,min_idx,_=find_path_rough(df,peak_space,sampling_dist)
        min_v_mm=[n*(10**-3) for n in min_v]
        tan,_=rough_approx(np.array(min_idx),np.array(min_v_mm))
        cos = np.sqrt(1/((tan**2)+1))
        print(tan)
        df=df.T
        for column in df.columns:
            df[column] = df[column]-tan*column*(10**3)
        df=df.T
        df.index=df.index/cos


        min_v,min_idx,peak_name=find_path_rough(df,peak_space,sampling_dist)

        min_info=make_info(df,min_idx)
        index_marker(df,min_idx,5)
        """
        df_trem=df.iloc[peak_index[0]-500:peak_index[-1]+500].copy()
        index_marker(df_trem,peak_name,5)
        """
        csp_name = peak_name[1:-1]

        csp_info = make_info(df,csp_name)
        index_marker(df,csp_name,5)
        return df,min_info,csp_info

def rough_approx(x,y):
    coeff = np.polyfit(x, y, 1)
    a,b = coeff
    return a,b
    


def make_info(df,index_li):
    space_li=[0]
    data_li = []
    for index in index_li:
        data_li.append(df.at[index,5])
    for i in range(len(index_li)-1):
        space_li.append(index_li[i+1]-index_li[i])
    info =pd.DataFrame(data_li,index=index_li,columns=["depth[um]"])
    info["space[mm]"]=space_li
    return info

def calc_trig(x=0,y=0,r=0):
    if not(r or x or y):
        print("not enough data for trig")
    elif(not(r)):#r==0
        r = math.sqrt(x**2+y**2)
        sin = y/r
        cos = x/r
        tan = y/x
    return sin,cos,tan 




def plot_3d(df,title):
    #conf#################
    LABEL_SIZE=10
    TITEL_SIZE=20
    x_label="traversal_dir[mm]"
    y_label="laser_dir[mm]"
    z_label="depth_dir[mm]"
    #######################
    X,Y = np.meshgrid(df.index,df.columns)
    ax = plt.subplot(projection = '3d')
    #ax.set_zlim(-1.55,-1.30)
    ax.plot_surface(X,Y,df.values.T,linewidth=0,cmap="bwr")#cmap="bwr"
    ax.set_title(title,fontsize=TITEL_SIZE)
    ax.set_xlabel(x_label,fontsize=LABEL_SIZE)
    ax.set_ylabel(y_label,fontsize=LABEL_SIZE)
    ax.set_zlabel(z_label,fontsize=LABEL_SIZE)

def calculate_ma(data_flame,window):
    data=data_flame
    for column in data_flame.columns:
        data[column]=data_flame[column].rolling(window,center=True).mean()
    return data

##find_peak##########################
def find_peak_XF(df,pach_space,sample_dist,path_count):
    dist = (pach_space/sample_dist)*0.7
    print("dist",dist)
    peak_df=pd.DataFrame()
    min_info_df=pd.DataFrame()
    min_info_df_idx=pd.DataFrame()
    find_col_li=[]
    unfind_col_li=[]
    csp_info_df=pd.DataFrame()
    csp_info_df_idx=pd.DataFrame()
    first_list_frag=True
    if type(df) == pd.core.series.Series:
        df=df.to_frame()
    for column in df.columns:
        min_info_el=[]
        min_info_el_idx=[]
        csp_info_el=[]
        csp_info_el_idx=[]
        df_col=df[column].to_frame()
        peak_li,_=find_peaks(df_col.iloc[:,0],distance=dist)
        if len(peak_li)  == path_count+1:
            first_list_frag=False
            find_col_li.append(column)
            peak_df[column]=peak_li
            last_peak_li= peak_li
        elif first_list_frag:
            print(len(peak_li))
            print("first")
            continue
        else:
            unfind_col_li.append(column)
            peak_li= last_peak_li
            peak_df[column]=peak_li

        for pach_num in range(len(peak_li)-1):
            #print(pach_num)
            min_info_el.append(find_min(df,column,peak_li[pach_num],peak_li[pach_num+1]))
            min_info_el_idx.append(find_min_idx(df,column,peak_li[pach_num],peak_li[pach_num+1]))
            csp_info_el.append(df_col.iat[peak_li[pach_num],0])
            csp_info_el_idx.append(df.index[peak_li[pach_num]])


        """
        for index in peak_li:
            if index:
                csp_idx.append(df_col.index[index])
                csp_v.append(df_col.iat[index,0])
            else:
                csp_idx.append(0)
                csp_v.append(0)
        csp_info=pd.DataFrame(csp_v,index=csp_idx)
        csp_df_li.append(csp_info)
        """
        #print(min_info_el)
    
        min_info_df[column]=min_info_el
        min_info_df_idx[column]=min_info_el_idx
        csp_info_df[column]=csp_info_el
        csp_info_df_idx[column]=csp_info_el_idx

    print("csp",csp_info_df)
    print("min",min_info_df)
    print("idx",min_info_df_idx)
    print(len(find_col_li))
    print(len(unfind_col_li))
    return [min_info_df,min_info_df_idx,csp_info_df,csp_info_df_idx]

def index_marker(df,index_df,column,int=False):
    print("index_mark")
    marker_df = df[column].to_frame().copy()
    marker_bottom=marker_df[column].min()
    marker_top=marker_df[column].max()
    marker_df["marker"]=marker_bottom
    if int:
        for index in index_df:
            if index == 0 :
                break
            else:
                marker_df.iat[index,-1]=marker_top
    elif isinstance(index_df,list):
        for index in index_df:
            marker_df.at[index,"marker"]=marker_top
    else:
        for index in index_df[column]:
            if index == 0 :
                break
            else:
                marker_df.at[index,"marker"]=marker_top
    marker_df.plot()
    
def find_min_idx(df,column,start_index,end_index):
    #print(df[column].iloc[start_index:end_index])
    return df[column].iloc[start_index:end_index].idxmin()

def find_min(df,column,start_index,end_index):
    return df[column].iloc[start_index:end_index].min()

##CROSS FEED func
def find_peak_FD(df,pach_space,sample_dist,sampl_len):
    max_peak_count=int(sampl_len/pach_space)+30
    print(sampl_len,max_peak_count)
    dist = (pach_space/sample_dist)*0.7
    print("dist",dist)
    peak_df=pd.DataFrame()
    min_info_df=pd.DataFrame()
    min_info_df_idx=pd.DataFrame()
    csp_info_df=pd.DataFrame()
    csp_info_df_idx=pd.DataFrame()
    find_col_li=[]
    unfind_col_li=[]
    if type(df) == pd.core.series.Series:
        df=df.to_frame()
    for column in df.columns:
        min_info_el=[]
        min_info_el_idx=[]
        csp_info_el=[]
        csp_info_el_idx=[]
        df_col=df[column].to_frame()
        #print(dist)
        #print(df_c.iloc[:,0])
        peak_li,_=find_peaks(df_col.iloc[:,0],distance=dist)
        #index_marker(df,peak_li,column,True)
        #print(peak_li)
        #ピークの数をそろえる（データフレーム結合のため）
        peak_count=len(peak_li)
        if peak_count <= max_peak_count:
            while len(peak_li)<=max_peak_count:
                peak_li=np.append(peak_li,0)
        find_col_li.append(column)
        peak_df[column]=peak_li
        
        
        for pach_num in range(len(peak_li)-1):
            if peak_li[pach_num+1]==0:#fill_num 0に達したらやめる
                min_info_el.append(0)
                min_info_el_idx.append(0)
                csp_info_el.append(0)
                csp_info_el_idx.append(0)
            else:
                min_info_el.append(find_min(df,column,peak_li[pach_num],peak_li[pach_num+1]))
                min_info_el_idx.append(find_min_idx(df,column,peak_li[pach_num],peak_li[pach_num+1]))
                csp_info_el.append(df_col.iat[peak_li[pach_num],0])
                csp_info_el_idx.append(df.index[peak_li[pach_num]])
        min_info_df[column]=min_info_el
        min_info_df_idx[column]=min_info_el_idx
        csp_info_df[column]=csp_info_el
        csp_info_df_idx[column]=csp_info_el_idx

    print("min",min_info_df)
    print("idx",min_info_df_idx)
    print(len(find_col_li))
    print(len(unfind_col_li))
    return [min_info_df,min_info_df_idx,csp_info_df,csp_info_df_idx]

def find_min_point_contact(df):
    target_df=df.df.copy().T
    XF_min_V_df,XF_min_IDX_df=find_peak_XF(target_df,df.path_spac,0.05,df.path_count)
    target_peak_FD= XF_min_V_df.T
    print(target_peak_FD)
    FD_min_V_df,FD_min_IDX_df=find_peak_FD(target_peak_FD,df.pach_space,df.sampling_distance,20)
    diff_df = FD_min_V_df.copy()
    
    for column in diff_df.columns:
        diff_df[column] = 2-(df.hight_surface-FD_min_V_df[column])
    diff_df=diff_df.applymap(lambda x: np.nan if x==2-df.hight_surface  else x)
    print(diff_df)
    print(diff_df.mean())
    return[[FD_min_V_df,XF_min_V_df],[FD_min_IDX_df,XF_min_IDX_df]]

#########################################








    


        




