import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
DataPath="${REBOOT_ROOT:-$PWD}/real_data" # path to data
Fast_data=["cell1-25.h5","cell2-35.h5","cell3-45.h5","cell7-25.h5","cell8-35.h5","cell9-45.h5","cell13-35.h5","cell14-45.h5","cell15-45.h5","cell16-45.h5","cell17-45.h5","cell18-45.h5","cell19-45.h5"]
Normal_data=["cell10-25.h5","cell11-35.h5","cell12-45.h5","cell4-25.h5","cell5-35.h5","cell6-45.h5"]
All_data=Fast_data+Normal_data

df=pd.read_hdf(DataPath+Fast_data[0][:-3]+".h5",key="df")
Ytrue=df["SOH"].to_numpy()


#to read one line (cycle) of a cell
i=0
plt.plot(df["Time[h]"][i][:min([len(df["Time[h]"][i]), len(df["I[A]"][i])])],df["I[A]"][i][:min([len(df["Time[h]"][i]), len(df["I[A]"][i])])])
plt.xlabel("Time [h]")
plt.ylabel("Current [A]")
plt.title("Current vs Time")
plt.show()

plt.plot(df["Time[h]"][i][:min([len(df["Time[h]"][i]), len(df["U[V]"][i])])],df["U[V]"][i][:min([len(df["Time[h]"][i]), len(df["U[V]"][i])])])
plt.xlabel("Time [h]")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs Time")
plt.show()

plt.plot(df["Time[h]"][i][:min([len(df["Time[h]"]), len(df["T1[°C]"])])],df["T1[°C]"][i][:min([len(df["Time[h]"]), len(df["T1[°C]"])])])
plt.xlabel("Time [h]")
plt.ylabel("Temperature [°C]")
plt.title("Temperature vs Time")
plt.show()

plt.plot(df["Time[h]"][i][:min([len(df["Time[h]"][i]), len(df["SOC"][i])])],df["SOC"][i][:min([len(df["Time[h]"][i]), len(df["SOC"][i])])])
plt.xlabel("Time [h]")
plt.ylabel("State of Charge")
plt.title("SOC vs Time")
plt.show()