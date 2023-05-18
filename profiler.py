import numpy as np
import time
import torch
import pynvml
from scipy import special,optimize
import csv, os, contextlib

def reset_power_cap(gpu_num):

    """Resets GPU power cap to the default setting.
    
        Parameters:
            gpu_num (int): index for which gpu to reset (default=0)"""
    
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
    default_cap = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(gpu_handle)
    pynvml.nvmlDeviceSetPowerManagementLimit(gpu_handle,default_cap)

def set_power_cap(gpu_num,cap):

    """Sets GPU power cap to the a chosen value in watts.
    
        Parameters:
            gpu_num (int): index for which gpu to reset (default=0)"""

    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
    pynvml.nvmlDeviceSetPowerManagementLimit(gpu_handle,cap*1000)


def power_cap_profiler(model,dataloader,gpu_num=0,train=False,loss_fn=None,optimizer=None,out_file=None):

    """Tests out different GPU power caps for PyTorch training or inference and suggests an optimal choice. 

        Parameters:
            model (torch.nn.Module): PyTorch model to be profiled\n
            dataloader (torch.utils.data.DataLoader): PyTorch dataloader containing the data for the model to be profiled on\n
            gpu_num (int): index for which gpu to profile (default=0)\n
            train (bool): whether the training or inference phase is to be profiled (True = training phase)\n
            loss_fn (torch.nn.module): PyTorch loss function that is used if train = True\n
            optimizer (torch.optim.Optimizer): PyTorch optimizer that is used if train = True\n
            out_file (string): Path to output file where profiler results will be saved. If equal to None then the results won't be saved to file\n
            """

    #check GPU is available and initialise profiler settings
    if torch.cuda.is_available():
        device = 'cuda:'+str(gpu_num)
    else:
        raise Exception('Profiler requires a CUDA device but none are available')
    model.to(device)
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
    default_cap = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(gpu_handle)
    power_caps = np.arange(1,0.25,-0.1)*default_cap #uses 8 different power caps (100%, 90%, 80%, 70%, 60%, 50%, 40% and 30% of the default cap)
    profile_time = 30 #the amount of time that each power cap is tested for
    profile_results = np.zeros((len(power_caps),4))

    #set model to training or inference mode
    if train:
        model.train()
    else:
        model.eval()
    
    #loop over the eight power caps
    for i,c in enumerate(power_caps):
        print("Testing:",str(round(c/1000))+"W")
        batch_count = 0
        start_time = time.time()
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
        pynvml.nvmlDeviceSetPowerManagementLimit(gpu_handle,round(c))

        #repeat execution until time is up
        while time.time() - start_time < profile_time:
            for batch,(data,labels) in enumerate(dataloader):
                if train:
                    optimizer.zero_grad()
                outputs = model(data.to(device))
                if train:
                    loss = loss_fn(outputs, labels.to(device))
                    loss.backward()
                    optimizer.step()
                batch_count += 1

                #stop current epoch if time runs out
                if time.time() - start_time >= profile_time:
                    break

        #calculate energy consumption in kWh
        energy_used = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) - start_energy)/3.6e9

        profile_results[i,0] = c
        profile_results[i,1] = batch_count
        profile_results[i,2] = energy_used
        profile_results[i,3] = energy_used/(batch_count**2) #proportional to ED^2P - used as minimisation objective but can be changed

    #parameterised model for predicting power cap performance
    def func(x,a,b,c,d,e,f,g): 
        return a*np.exp(b*x-c) + d*special.expit(e*x-f) +g
    
    optimised = False
    attempt = 0
    while not optimised:
        attempt += 1

        #if model cannot be fitted then the best tested power cap is chosen
        if attempt > 100:
            print("MODEL FITTING FAILED - choosing optimal powercap from tested options")
            optimal_cap = profile_results[np.argmin(profile_results[:,3]),0]
            optimised = True
        else:
            #randomly chooses initial guess for curve_fit
            p0 = np.random.randn(7)
            try:
                #uses least squares to find coefficient values from data and initial guess - context managers are used to hide output
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        params, params_covariance = optimize.curve_fit(func, power_caps/default_cap, profile_results[:,3], p0=p0)
                #power cap predicition model using found coefficients
                def fitted_func(x):
                    return params[0]*np.exp(params[1]*x-params[2]) + params[3]*special.expit(params[4]*x-params[5]) + params[6]
                
                #minimises the fitted function to find optimal power cap)
                error = np.mean(np.abs((profile_results[:,3] - fitted_func(power_caps/default_cap)) / profile_results[:,3])) * 100

                #accepts model fit if mean percentage error is less than 5%
                if error < 5:
                    optimised = True
                    #sets optimal cap to min of fitted function
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            optimal_cap = optimize.fmin(fitted_func,np.array([0]))[0]*default_cap
            except:
                continue
    
    #convert chosen power cap from mW to W
    optimal_cap = round(optimal_cap/1000)
    
    #output profiler results if an output file is given
    if out_file is not None:
        with open(out_file,'w+') as file:
            writer = csv.writer(file)
            writer.writerows(profile_results)
            writer.writerow(["Modeled Optimal Cap:",optimal_cap])

    #reset GPU power cap back to default
    pynvml.nvmlDeviceSetPowerManagementLimit(gpu_handle,default_cap)

    return optimal_cap