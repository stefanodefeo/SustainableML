### This repo contains the custom energy measurement tool and power cap profiler from Stefano De Feo's master's thesis: "Towards Sustainable Machine Learning Through GPU Power Capping"

# Requirements:

To install all dependencies, use ```pip install -r requirements.txt```.

# Measurement Tool:

The tool has three different measurement modes which are contained in three seperate classes:
1. ```EnergyMeasurementTool``` - Measures the energy consumption and execution time of code snippets.
2. ```AsynchronousMeasurementTool``` - Runs as a seperate process and takes readings at a given interval for the following metrics: energy (CPU & GPU), utilisation (CPU & GPU), temperature (CPU & GPU), power (GPU only).
3. ```LayerwiseMeasurementTool``` - Measures the energy consumption per layer of a PyTorch neural network based on each layer's execution time.

## Installation:

To use the measurement tool simply import the class for the specific mode that is required. For example:

```python
from measurement import EnergyMeasurementTool
from measurement import AsynchronousMeasurementTool
from measurement import LayerwiseMeasurementTool 
```

## Usage:

### ```EnergyMeasurementTool```:

1. Create an ```EnergyMeasurementTool``` object.
2. Call ```start()``` method to begin measuring a code snippet.
3. Run code to be measured.
4. Call ```stop()``` method to finish measuring.
5. On program exit, all results will be saved to the output file given to ```EnergyMeasurementTool```'s constructor. The columns and units of the csv are: 

    elapsed time (s) | CPU energy (kWh) | GPU energy (kWh)

### Example:

```python
tracker = EnergyMeasurementTool(out_file='test_energy_measurement_tool.csv',gpu_num=0)
for epoch in range(num_epochs):
    tracker.start()
    train()
    test()
    tracker.stop()
```

### ```AsynchronousMeasurementTool```:

1. Create an ```AsynchronousMeasurementTool``` object.
2. Call ```start()``` method with *interval* argument to begin taking asynchronous measurements every *interval* seconds.
3. Run code to be profiled.
4. Call ```stop()``` method to finish taking measurements.
5. On program exit, all results will be saved to the output file given to ```AsynchronousMeasurementTool```'s constructor. The columns and units of the csv are: 

    elapsed time (s) | CPU energy (kWh) | CPU utlisation (%) | CPU temp (&deg;C) | GPU energy (kWh) | GPU utilisation (%) | GPU temp (&deg;C) | GPU power (W)

### Example:

```python
tracker = AsynchronousMeasurementTool(out_file='test_asynchronous_measurement_tool.csv',gpu_num=0)
tracker.start(1)
for epoch in range(num_epochs):
    train()
    test()
tracker.stop()
```

### ```LayerwiseMeasurementTool```:

1. Create an ```LayerwiseMeasurementTool``` object.
2. Call ```start()``` method with *model* argument to obtain a measurable model and begin the overrall energy consumption measurements that are used to estimate layerwise consumption. This must be done immediately before the training/inference loop.
3. Run training or inference code to measure layerwise consumption. **NOTE:** training and inference cannot be run together.
3. Call ```stop()``` method to finish taking overrall energy measurements and use this to estimate layerwise consumption.
4. Once ```stop()``` has been called, all results will be saved to the output file given to ```LayerwiseMeasurementTool```'s constructor. This consists of the average energy consumption in joules for each layer in the model.

### Example:

```python
tracker = LayerwiseMeasurementTool('test_layerwise_measurement_tool.csv',0)
model = tracker.start(model)
for epoch in range(num_epochs):
    train()
tracker.stop()
```

# Power Cap Profiler

- The profiler is used to find an optimal GPU power cap for PyTorch neural network training or inference.
- It tests out eight different power caps for a short duration and suggests an optimal choice in terms of minimising the ED<sup>2</sup>P.
- The chosen power cap may not be one of the eight that are tested because a prediction model is used to estimate the performance of untested power caps.
- Using the chosen power cap for neural network training or inference will produce energy savings for minimal increases in execution time.
- Two utility functions are also included to set GPU power caps without having to interact with the NVML API directly (```set_power_cap``` and ```reset_power_cap```).

## Installation:

To use the power cap profiler and utlity funcions, simply import the required functions into your python program. For example:

```python
from profiler import power_cap_profiler
from profiler import set_power_cap
from profiler import reset_power_cap
```

## Usage:

1. Call ```power_cap_profiler``` with arguments that will be used in the real PyTorch training/inference run. For example, the dataloader supplied must be the same dataloader that will be used in the full program execution.
2. Call ```set_power_cap``` with the chosen gpu index and the optimal power cap that was returned from the profiler.
3. Execute PyTorch training or inference run and receive energy savings compared to the default cap with minimal increase in runtime.
4. Call ```reset_power_cap``` to reset the GPU back to its default power cap.

### Example:

```python
#Profiling
model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
optimal_cap = power_cap_profiler(gpu_num=0,model=model,dataloader=trainloader,train=True,loss_fn=criterion,optimizer=optimizer,out_file='test_power_cap_profiler.csv')

#Real training run
model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
set_power_cap(gpu_num=0,cap=optimal_cap)
train()
reset_power_cap(gpu_num=0)
```