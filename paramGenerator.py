'''
© Author: Saunak Saha
Iowa State University
'''
import sys
import numpy as np


#accepting command-line arguments
variable = sys.argv[1]
start_val = float(sys.argv[2])
end_val = float(sys.argv[3])
step_val = float(sys.argv[4])

#reading values fromm default file
f = open('parameters_default.txt',"r")
lines = f.readlines()
params = []
for x in lines:
    params.append(x.split())
f.close()
params = np.array(params)



#fixing variable in question 
varIndex=90
for m in range (0,23):
    if(variable==params[m,0]):
        varIndex=m
        break


#generating new files with range of values as required
params2 = params
values = np.arange(start_val, end_val+step_val , step_val)
i=0
for n in values: 
    params2[varIndex,2]=n
    np.savetxt('parameters_'+variable+'_'+str(i)+'.txt', params2, fmt='%s')
    i+=1

#calculating number of simulations required 
simNumber = (end_val - start_val)/step_val+1
print("Put simNumber as: "+str(simNumber));
    
