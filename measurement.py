import time
import pynvml
import csv
import os
import atexit
import torch.nn as nn
import torch
import numpy as np
import psutil
import multiprocessing

class EnergyMeasurementTool():

    """ A class to measure the energy consumption and execution time of code snippets. """


    def __init__(self,out_file,gpu_num=0) -> None:

        """Initialises the tool.

        Parameters:
            out_file (string): path to output file\n
            gpu_num (int): index for which gpu to monitor (default=0)"""
        
        self.times = []
        self.cpu_energy = []
        self.gpu_energy = []
        self.out_file = out_file
        open(self.out_file,'w+').close()
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
        atexit.register(self.flush)
        self.max_cpu_energy = int(open('/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj','r').read())
    
    def start(self):

        """Starts measuring."""

        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', "r") as f:
            self.cpu_start = int(f.read())
        self.gpu_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
        self.start_time = time.time()

    def stop(self):

        """Stops measuring and calculates elapsed time and gpu and cpu energy consumption since start was called."""
        
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', "r") as f:
            cpu_end = int(f.read())
        if cpu_end < self.cpu_start:
            cpu_end += self.max_cpu_energy
        self.cpu_energy.append((cpu_end - self.cpu_start)/3.6e12)
        self.gpu_energy.append((pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle) - self.gpu_start)/3.6e9) #Both energy readings converted to kWh
        self.times.append(time.time()-self.start_time)

    def flush(self):

        """Writes all recorded times and energy consumptions to the output file in csv format. This function is automatically called on program exit.
        
        The order of the csv columns is: time | cpu energy | gpu energy"""
       
        with open(self.out_file,'w+') as f:
            writer = csv.writer(f)
            writer.writerows(zip(self.times,self.cpu_energy,self.gpu_energy))
                
class AsynchronousMeasurementTool():

    """ A class to asynchronously measure many CPU and GPU metrics:
      
    CPU metrics: energy consumption, utilisation, temperature
    GPU metrics: energy consumption, utilisation, temperate, power """
     
    def __init__(self,out_file,gpu_num=0) -> None:
        
        """Initialises the tool.

        Parameters:
            out_file (string): path to output file\n
            gpu_num (int): index for which gpu to monitor (default=0)"""
        
        self.times = []
        self.cpu_energy = []
        self.cpu_utilisation = []
        self.cpu_temp = []
        self.gpu_energy = []
        self.gpu_utilisation = []
        self.gpu_temp = []
        self.gpu_power = []
        self.out_file = out_file
        self.gpu_num = gpu_num
        open(self.out_file,'w+').close()
        self.max_cpu_energy = int(open('/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj','r').read())
    
    def start(self,interval=1):

        """Starts the tool.

        Parameters:
            interval (float): time in seconds between each reading (default=1)"""
        
        conn_recv, self.conn_send = multiprocessing.Pipe(False)
        p = multiprocessing.Process(target=self._run, args=(conn_recv, interval))
        p.start()
    
    def stop(self):

        """Stops the tool."""

        self.conn_send.send(True)

    def _run(self,stop_conn,interval):
        #Called in a seperate process by start. Calls read every interval seconds until stop is called.
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_num)
        self.start_time = time.time()
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', "r") as f:
            self.start_cpu_energy = int(f.read())
        self.start_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
        while not stop_conn.poll():
            self.read()
            time.sleep(interval)
        self.flush()

    def read(self):
        
        """Takes a reading of all metrics. Called by run at its specified interval"""
        
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', "r") as f:
            cpu_end = int(f.read())
        if cpu_end < self.start_cpu_energy:
            cpu_end += self.max_cpu_energy
        self.cpu_energy.append((cpu_end - self.start_cpu_energy)/3.6e12)
        self.cpu_utilisation.append(psutil.cpu_percent(0.1))
        with open('/sys/class/thermal/thermal_zone3/temp', "r") as f:
            self.cpu_temp.append(int(f.read())/1000)
        self.gpu_energy.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)-self.start_gpu_energy)
        self.gpu_utilisation.append(pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu)
        self.gpu_temp.append(pynvml.nvmlDeviceGetTemperature(self.gpu_handle,0))
        self.gpu_power.append(pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)/1000)
        self.times.append(time.time()-self.start_time)
        

    def flush(self):
        
        """Writes all recorded mertics to the output file in csv format. This function is automatically called when run recieves a stop signal and returns.
        
        The order of the csv columns is: time | cpu energy | cpu utlisation | cpu temp | gpu energy | gpu utilisation | gpu temp | gpu power"""
        
        with open(self.out_file,'a') as f:
            writer = csv.writer(f)
            writer.writerows(zip(self.times,self.cpu_energy,self.cpu_utilisation,self.cpu_temp,self.gpu_energy,self.gpu_utilisation,self.gpu_temp,self.gpu_power))
    

class MeasureLayer():

    #A class to measure an individual layer's energy consumption. Called by LayerwiseMeasurmentTool. Should not be called directly

    def __init__(self,layer_type) -> None:
        self.forward_times = []
        self.backward_times = []
        self.layer_type = layer_type

    def start_forward_hook(self,module,input):
        #records the start time of the forward pass of the layer
        self.start_forward_time = time.time()
    
    def start_backward_hook(self,module,grad_input):
        #records the start time of the backward pass of the layer
        self.start_backward_time = time.time()

    def stop_forward_hook(self,module,input,output):
        #saves the execution time for a forward pass of the layer
        end_time = time.time()
        self.forward_times.append(end_time-self.start_forward_time)
    
    def stop_backward_hook(self,module,grad_input,grad_output):
        #saves the execution time for a backward pass of the layer
        end_time = time.time()
        self.backward_times.append(end_time-self.start_backward_time)

    def finalise(self,average_power):
        #converts execution time to energy and calculates average energy consumption of the layer
        self.forward_energies = [t*average_power for t in self.forward_times]
        self.backward_energies = [t*average_power for t in self.backward_times]
        self.total_energies = [self.forward_energies[i] + self.backward_energies[i] for i in range(len(self.forward_energies))]
        return self.layer_type,np.mean(self.total_energies)
    

class LayerwiseMeasurementTool():

    """A class to measure the energy consumption per layer of a PyTorch DNN."""

    def __init__(self,out_file,gpu_num=0) -> None:

        """Initialises the tool.

        Parameters:
            out_file (string): path to output file\n
            gpu_num (int): index for which gpu to monitor (default=0)"""
        
        self.gpu_num = gpu_num
        self.out_file = out_file
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
        self.max_cpu_energy = int(open('/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj','r').read())
    
    def start(self,model):

        """Creates model with layerwise measuring capability and starts overall energy measurements. Should be called immediately before training loop.
        
        All layers to be measured should inherit from torch.nn.Module and be instatiated in the network's __init__ method 
        
        Parameters:
            model (torch.nn.Module): PyTorch model to be converted into a measurable version."""
        
        model, self.layer_trackers = LayerwiseMeasurementTool.create_measurable_model(model)
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', "r") as f:
            self.start_cpu_energy = int(f.read())
        self.start_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
        self.start_time = time.time()
        return model
    
    def stop(self):

        """Stops measuring total energy and uses this to convert layerwise measurements to real energy readings. Also outputs data to the given file."""
        
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', "r") as f:
            cpu_end = int(f.read())
        if cpu_end < self.start_cpu_energy:
            cpu_end += self.max_cpu_energy
        cpu_total_energy = (cpu_end - self.start_cpu_energy)/1e6  #Both energy readings converted to joules
        gpu_total_energy = (pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle) - self.start_gpu_energy)/1000
        total_time = time.time() - self.start_time
        total_energy = cpu_total_energy + gpu_total_energy
        average_power = total_energy/total_time
        layer_energies = [l.finalise(average_power) for l in self.layer_trackers]
        with open(self.out_file,'w+') as f:
            writer = csv.writer(f)
            writer.writerows(layer_energies)

    @staticmethod
    def create_measurable_model(model):
        #Creates a measurable model from a PyTorch model by inserting forward and backward hooks.
        num = 0
        trackers = []
        for layer in nn.ModuleList(model.modules())[1:]:
            if not isinstance(layer, nn.Sequential) and torch.nn.__name__ in layer.__module__:
                tracker = MeasureLayer(type(layer).__name__)
                layer.register_forward_pre_hook(tracker.start_forward_hook)
                layer.register_forward_hook(tracker.stop_forward_hook)
                layer.register_full_backward_pre_hook(tracker.start_backward_hook)
                layer.register_full_backward_hook(tracker.stop_backward_hook)
                trackers.append(tracker)
                num+=1
        return model, trackers