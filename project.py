import pandas as pd
import numpy as np
import seaborn as sns


path_cus = "cus_info.csv"
path_act = "act_info.csv"
path_iem = "iem_info.csv"
path_trd_kr = "trd_kr.csv"
path_trd_oss = "trd_oss.csv"

'''
Header Information
df_cus = pd.read_csv("cus_info.csv")
df_act = pd.read_csv("act_info.csv")
df_iem = pd.read_csv("iem_info.csv")
df_trd_kr = pd.read_csv("trd_kr.csv")

고객번호,성별,연령대,주소(시도),고객등급,고객투자성향
df_cus.columns = ['cus_id', 'sex_dit_cd', 'cus_age', 'zip_ctp_cd', 'tco_cus_grd_cd', 'ivs_icn_cd']
계좌번호,고객번호,계좌개설년월
df_act.columns = ['act_id', 'cus_id', 'act_opn_ym']
종목코드,종목영문명,종목한글명
df_iem.columns = ['iem_cd', 'iem_eng_nm', 'iem_krl_nm']
계좌번호,주문날짜,주문순서,주문접수시간대,최종체결시간대,종목코드,매매구분코드,체결수량,체결가격,주문매체구분코드
df_trd_kr.columns = ['act_id', 'orr_dt', 'orr_ord', 'orr_rtn_hur', 'lst_cns_hur', 'iem_cd', 'sby_dit_cd', 'cns_qty', 'orr_pr', 'orr_mdi_dit_cd']
'''



def hist():
	import matplotlib.pyplot as plt
	data1 = pd.read_csv("merged_s100.csv")
	data1.hist(bins=10, figsize=(12, 10))
	plt.show()


def merge_csv(path1,path2,label_name,na_flag,result_path):	#label_name: 병합기준컬럼 , na_flag: True 일때 dropna
	import pandas as pd
	data1 = pd.read_csv(path1)
	data2 = pd.read_csv(path2)
	result = pd.merge(data1, data2, how = 'left', on = label_name)
	if na_flag:
		result.dropna()
	if not result_path:
		return result
	else:
		result.to_csv(result_path, index=False)

def sampling_csv(path1,percentile,seed,result_path):	#percentile: 샘플링 비율 , seed: 랜덤값
	from sklearn.model_selection import train_test_split
	data1 = pd.read_csv(path1)
	#result, x_test = train_test_split(data1, train_size=percentile, random_state=seed)
	result, dummy = train_test_split(data1, train_size=percentile, random_state=seed)
	result.to_csv(result_path, index=False)
	if not result_path:
		return result
	else:
		result.to_csv(result_path, index=False)

def split_csv(path1,label_name,prefix):
	data1 = pd.read_csv(path1)
	data1 = data1.sort_values(by=label_name)
	counts = data1[label_name].value_counts()
	counts = counts.sort_index()
	for i in counts.index:
		temp = data1[data1[label_name] == i]
		temp.to_csv(prefix+"_"+str(i)+".csv", index=False)


def split_data_by_percentile(data1, label_name, percentile1, percentile2, how): #percentile: 비율 , how: 'q'/'c'  전체 데이터 수에 대해서/누적값에 대해서
	label_data = data1[label_name]
	label_data = label_data.sort_values(ascending=False)
	if how=='ca':
		label_data = label_data[::-1]
	percentile1, percentile2 = percentile1*0.01, percentile2*0.01

	if how == 'q': #Qunatile Function
		return label_data[(label_data.values >= label_data.quantile(q=percentile1)) & (label_data.values <= label_data.quantile(q=percentile2))]
	if how == 'ca' or how == 'cd': #Accumulative Ascending & Descending
		cum_data = label_data.cumsum()
		target1 = cum_data.iloc[-1] * percentile1
		target2 = cum_data.iloc[-1] * percentile2
		return label_data[(cum_data.values >= target1) & (cum_data.values <= target2)]
	print("Invaled Tag ('q' or 'ca' or 'cd')")
	return None


def analysisTemp():
	import matplotlib.pyplot as plt
	import numpy as np
	data1 = pd.read_csv("merged_s100.csv")
	data1['money'] = data1['cns_qty']*data1['orr_pr']
	df = split_data_by_percentile(data1,'money',0,80,'cd')
	print(df.shape[0] / data1.shape[0])
	sns.lineplot(x=np.arange(len(df)), y=df.values)
	plt.show()
	df = split_data_by_percentile(data1,'money',0,80,'ca')
	print(df.shape[0] / data1.shape[0])

#Prefix Custom Type A
def merge_typeA():
	#prefix Depth 1 : df_act + df_cus
	merge_csv(path_act,path_cus,'cus_id',False,"merged_act_cus.csv")

	#prefix Depth 2 : sampling trd_kr data (size = 0.05, seed = 100)
	sampling_csv(path_trd_kr,0.05,100,"trd_kr_s100.csv")

	#prefix Depth 3 : trd_kr_sample + df_act + df_cus
	merge_csv("trd_kr_s100.csv","merged_act_cus.csv",'act_id',False,"merged_s100.csv")

def merge_typeZ():
	merge_csv(path_trd_kr,"merged_act_cus.csv",'act_id',False,"merged_TKAC.csv")


def swap_column(df, i, j): #change i <-> j
	temp = df.columns.tolist()
	temp[i], temp[j] = df.columns[j], df.columns[i]
	return df[temp]
	


def analysis1(): #연령별 등급 분포
	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt

	df1 = pd.read_csv("cus_info.csv")
	#df2 = pd.read_csv("merged_TKAC.csv") 
	df2 = pd.read_csv("merged_s100.csv") #code test

	##age_grd[0]['04'][45] -> 인덱싱 접근법
	cus_by_grd = df1.groupby(['cus_age','tco_cus_grd_cd']).size().unstack(fill_value=0)
	deal_by_grd = df2.groupby(['cus_age','tco_cus_grd_cd']).size().unstack(fill_value=0)
	result = [cus_by_grd, deal_by_grd, (deal_by_grd/cus_by_grd).fillna(0)]
	
	cus_by_grd[cus_by_grd.index <= 35] #분포 확인
	deal_by_grd[deal_by_grd.index <= 35] #분포 확인

	target = [20,25,30,35]
	##target = age_grd.index #original
	

	#Graph1
	fig = plt.figure(figsize=(14, 3))
	for i in range(3):
		ax = fig.add_subplot(1, 3, (i+1))
		for j in target:
			x = result[i%3].columns.tolist()
			y = result[i%3][result[i%3].index == j].values.reshape(-1)
			sns.lineplot(x=x, y=y, label=j, ax=ax)
	plt.show()


	#Graphe2 - Prefixed Version
	result[2]['01'][25] = result[2]['01'].mean()
	fig = plt.figure(figsize=(14, 3))
	for i in range(3):
		ax = fig.add_subplot(1, 3, (i+1))
		for j in target:
			x = result[i%3].columns.tolist()
			y = result[i%3][result[i%3].index == j].values.reshape(-1)
			sns.lineplot(x=x, y=y, label=j, ax=ax)
	plt.show()		


#	sns.lineplot(x=x, y=y, label=i, ax=ax2)
	#sns.lineplot(data=age_grd.values, x=x1, y=y1 ) #x="year", y="passengers" hue="cus_age"
	#sns.heatmap(age_grd,cmap='Blues', annot=True, fmt="")


#analysis1()


def analysis2():
	import matplotlib.pyplot as plt
	import seaborn as sns
	#data1 = pd.read_csv("merged_TKAC.csv")
	#data1 = pd.read_csv("prefix/age_45.csv")
	data1 = data1 = pd.read_csv("prefix/age_20.csv")
	data1['money'] = data1['cns_qty']*data1['orr_pr']

	msss = data1['money'].sort_values(ascending=False)
	msss.to_csv("temp.csv",index=False)


	temp = pd.DataFrame(data1.groupby('cus_id')['money'].sum())
	temp = temp.sort_values('money',ascending=False)
	#temp.to_csv("temp.csv",index=False)

	fig = plt.figure(figsize=(10, 8))
	ax = [fig.add_subplot(2, 2, 1),
		fig.add_subplot(2, 2, 2),
		fig.add_subplot(2, 2, 3),
		fig.add_subplot(2, 2, 4)]

	percentile = [40,80,90,100]

	for i in range(4):

		temp2 = temp[temp['money'] <= np.percentile(temp['money'],percentile[i])]
		x1 = np.arange(len(temp2))
		sns.lineplot(x=x1, y=temp2['money'],ax=ax[i])
	#temp2.to_csv("temp2.csv",index=False)
	#temp['money'].plot()
	plt.show()


	#new_data = data1[label_name].np.percentile(percentile)
	#print(new_data)
#x = split_data_by_percentile(data1,'money',20,100,'q')
#y = split_data_by_percentile(data1,'money',90,'c')
#print(x.shape[0] / data1.shape[0] * 100)
#print(y.shape[0] / data1.shape[0] * 100)