import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
# GROUND TRUTH DEFORMATION GRADIENT 
##########################################################################################
if False:
	max_contract = 0.15
	val_list = []
	for kk in range(0,5):  val_list.append(0)
	for kk in range(0,20): val_list.append(-1.0*max_contract*np.sin(kk/40*np.pi*2))
	for kk in range(0,5):  val_list.append(0)
	for kk in range(0,20): val_list.append(-1.0*max_contract*np.sin(kk/40*np.pi*2))
	for kk in range(0,5):  val_list.append(0)
	for kk in range(0,20): val_list.append(-1.0*max_contract*np.sin(kk/40*np.pi*2))
	for kk in range(0,5):  val_list.append(0)


val_list_A = []
v = .005
v2 = 40
for kk in range(0,15): val_list_A.append(kk*v)
for kk in range(0,5):  val_list_A.append(15*v) 
for kk in range(0,15): val_list_A.append(15*v - kk*v)
for kk in range(0,15): val_list_A.append(kk*v)
for kk in range(0,5):  val_list_A.append(15*v) 
for kk in range(0,15): val_list_A.append(15*v - kk*v)
for kk in range(0,5):  val_list_A.append(15*v) 
for kk in range(0,5):  val_list_A.append(kk*v)


val_list_B = [] 
for kk in range(0,int(v2*2)): val_list_B.append(np.sin(kk/20)*.1)

##########################################################################################
# COMPUTED DEFORMATION GRADIENT 
##########################################################################################
x_pos = np.loadtxt('synthetic_data_3/synthetic_data_3_GT_x_pos_array.txt')
y_pos = np.loadtxt('synthetic_data_3/synthetic_data_3_GT_y_pos_array.txt')

num_sarc = x_pos.shape[0]
num_time = x_pos.shape[1]
num_vec = int((num_sarc * num_sarc - num_sarc) / 2.0)

Lambda_list = []
for tt in range(0,num_time):
	Lambda = np.zeros((2,num_vec))
	ix = 0
	for kk in range(0,num_sarc):
		for jj in range(kk+1,num_sarc):
			x_vec = x_pos[kk,tt] - x_pos[jj,tt]
			y_vec = y_pos[kk,tt] - y_pos[jj,tt]
			Lambda[0,ix] = x_vec
			Lambda[1,ix] = y_vec 
			ix += 1 
	
	Lambda_list.append(Lambda)

F_list = [] 
F11_list = []
F22_list = [] 
F12_list = []
F21_list = [] 
for tt in range(0,num_time):
	Lambda_0 = Lambda_list[0]
	Lambda_t = Lambda_list[tt]
	term_1 = np.dot( Lambda_t , np.transpose(Lambda_0) )
	term_2 = np.linalg.inv( np.dot( Lambda_0 , np.transpose(Lambda_0) ) )
	F = np.dot(term_1 , term_2)
	F_list.append(F)
	F11_list.append(F[0,0] - 1.0)
	F22_list.append(F[1,1] - 1.0)
	F12_list.append(F[0,1])
	F21_list.append(F[1,0])
	
##########################################################################################
# PLOT 
##########################################################################################
plt.figure()
plt.plot(F11_list,'r--',linewidth=5, label='F11 recovered')
plt.plot(F22_list,'g--',linewidth=4, label='F22 recovered')
plt.plot(F12_list,'c:',label='F12 recovered')
plt.plot(F21_list,'b:',label='F21 recovered')
#plt.plot(val_list,'k--',label='ground truth')
plt.plot(val_list_A,'k--',label='ground truth group A')
plt.plot(val_list_B,'k:',label='ground truth group B')
plt.legend()
plt.savefig('synthetic_data_3/recovered_F')

	
