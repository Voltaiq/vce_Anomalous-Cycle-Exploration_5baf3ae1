import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display, Markdown
import voltaiq_studio as vs
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interactive,fixed

from sklearn.neighbors import LocalOutlierFactor


def filter_test_record_by_name(name, trs):
    return [t for t in trs if name.lower() in t.name.lower()]

def filter_test_record_by_names(names, trs):
    ''' 
    Filter test records by a list of names.
    Return list of test record objects
    '''
    return [t.name for t in trs if all(x.lower() in t.name.lower() for x in names)]

def filter_test_record_by_one_name(name, trs):
    ''' 
    Filter test records by a single keyword/name.
    Return list of test record objects
    '''
    return [t.name for t in trs if name.lower() in t.name.lower()] 

def filter_tr_by_name_exclusion(name,exclude, trs):
    ''' 
    Filter test records by a single keyword/name. 
    Exclude any test records from the list which are in the exclude list.
    Return list of test record objects
    '''
    return [t.name for t in trs if (name.lower() in t.name.lower() and all(x.lower() not in t.name.lower() for x in exclude))]

# (optional) remove first n & last n cycles (useful for removal of incomplete cycles)
def remove_firstn_and_lastn_cycles(cycle_df, n_start, n_end):
    n_start_cycles = []
    n_end_cycles = []
    if n_start:
        n_start_cycles = list(range(n_start))
    if n_end:
        last_cycle = int(cycle_df.iloc[-1]["Cycle Number"])
        n_end_cycles = list(range(last_cycle, last_cycle-n_end, -1))
    return cycle_df[~cycle_df["Cycle Number"].isin(n_start_cycles + n_end_cycles)]

# find cycles with nan values for a given cycle stat 'trace'
def find_nan_cycles(cycle_df, trace):
    return cycle_df[cycle_df[trace].isna()], cycle_df.dropna(subset=[trace])

# scale values from 0-1
def scale_data(cycle_data):
    scaler = MinMaxScaler() 
    df = scaler.fit_transform(cycle_data)
    return pd.DataFrame(df, columns = cycle_data.columns)

# use LocalOutlierFactor with default arguments: n_neighbors = 20 & default=’minkowski’ & ...
def run_LOF(scaled_df, cycle_stats):
    # start with default parameters
    default_LOF = LocalOutlierFactor()
    prediction = default_LOF.fit_predict(scaled_df)
    return cycle_stats.iloc[np.where(prediction == -1)]

# plot cycle data and anomalies
def plot_data(cycle_stats, anomalies, trace):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycle_stats.index, y=cycle_stats[trace], mode="lines", name="Anomalous "+trace, visible='legendonly'))
    cycle_stats = cycle_stats.drop(anomalies.index)
    fig.add_trace(go.Scatter(x=cycle_stats.index, y=cycle_stats[trace], mode="lines", name=trace))
    fig.update_layout(
        title=trace,
        xaxis_title="Cycle Number",
        yaxis_title=trace,
    )
    fig.show()

# plot current & potential for a given test & cycle pair
def plot_cycle_timeseries(test, cycle):
    reader = test.make_time_series_reader()
    reader.add_trace_keys('h_current', 'h_potential', 'h_datapoint_time')
    reader.add_info_keys('i_cycle_num')
    reader.filter_cycle_list([cycle])
    df = reader.read_pandas()
    df["timestamp"] = pd.to_datetime(df['h_datapoint_time'],unit='ms')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["h_potential"], mode="lines", name="Potential"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["h_current"], mode="lines", name="Current", yaxis="y2"))
    fig.update_layout(
        title=f"Cycle Number {cycle}",
        xaxis_title="Test Timestamp",
        yaxis_title="Potential (V)",
        yaxis2=dict(
            title="Current (A)",
            anchor="x",
            overlaying="y",
            side="right",
        ),
    )
    fig.show()
    
    
std_train_datasets = ['Severson2019 - All (LFP)','Severson2019 - Train (LFP)',
                      'Severson2019 - Test (LFP)','Severson2019 - Test2 (LFP)','Attia2020 (LFP)',
                      'Devie2018 (NMC/LCO)','Wei2011 (LCO)','Juarez-Robles2020 (NCA)',
                      'Weng2021 (NMC)','Custom']

def get_standard_datasets(dataset, trs):
    # Severson train_test_test2 split
    # list of names to filter
    if dataset in ['Severson2019 - All (LFP)','Severson2019 - Train (LFP)',
                      'Severson2019 - Test (LFP)','Severson2019 - Test2 (LFP)']:
        dates = ['2017-05-12','2017-06-30']
        train_channels_512 = ['CH2_','CH5_','CH7_','CH9_','CH14_','CH16_',
                              'CH24_','CH18_','CH12_','CH25_','CH33_','CH27_','CH35','CH29','CH39',
                              'CH37','CH41','CH43','CH45','CH47']
        train_channels_630 = ['CH9_','CH11_','CH13_','CH15_','CH27_','CH29_',
                              'CH17_','CH19_','CH21','CH23','CH7_','CH24','CH26','CH32','CH34',
                              'CH36','CH38','CH40','CH42','CH44','CH46']
        train = []
        for ch in train_channels_512:
            train.append([dates[0],ch])
        for ch in train_channels_630:
            train.append([dates[1],ch])

        test_channels_512 = ['CH1_','CH3_','CH6_','CH10_','CH20','CH15_','CH23_',
                             'CH17_','CH11_','CH32_','CH26_','CH34_','CH28_','CH36_','CH30_','CH40_',
                             'CH38_','CH42_','CH44_','CH46_','CH48_']
        test_channels_630 = ['CH10_','CH12_','CH14_','CH16_','CH28_','CH30','CH18_',
                             'CH20_','CH22_','CH4_','CH8_','CH25_',
                             'CH31_','CH33_','CH35_','CH37_','CH39_','CH41_','CH43_','CH45_','CH47_','CH48_']
        test = []
        for ch in test_channels_512:
            test.append([dates[0],ch])
        for ch in test_channels_630:
            test.append([dates[1],ch])

        test2_names = 'batch8'
        remove_test2 = ['2018-04-12_batch8_CH33','2018-04-12_batch8_CH46','2018-04-12_batch8_CH7',
                        '2018-04-12_batch8_CH41','2018-04-12_batch8_CH6','2018-04-12_batch8_CH12']
    
        tr_list_train = [filter_test_record_by_names(name, trs) for name in train]
        tr_list_train_flat = [item for sublist in tr_list_train for item in sublist]
        tr_list_test = [filter_test_record_by_names(name, trs) for name in test]
        tr_list_test_flat = [item for sublist in tr_list_test for item in sublist]
        tr_list_test2 = filter_tr_by_name_exclusion(test2_names, remove_test2, trs)

    if dataset == 'Severson2019 - All (LFP)':
        return tr_list_train_flat + tr_list_test_flat + tr_list_test2
    elif dataset == 'Severson2019 - Train (LFP)':
        return tr_list_train_flat
    elif dataset == 'Severson2019 - Test (LFP)':
        return tr_list_test_flat
    elif dataset == 'Severson2019 - Test2 (LFP)':
        return tr_list_test2
    elif dataset == 'Attia2020 (LFP)':
        return filter_test_record_by_one_name("batch9",trs)
    elif dataset == 'Devie2018 (NMC/LCO)':
        return filter_test_record_by_names(["HNEI","timeseries"], trs)
    elif dataset=='Wei2011 (LCO)':
        return filter_test_record_by_one_name("CALCE", trs)
    elif dataset == 'Juarez-Robles2020 (NCA)':
        return filter_test_record_by_one_name("UL-PUR", trs)
    elif dataset =='Weng2021 (NMC)':
        return filter_test_record_by_names(["UM","Cycling"], trs)

# widget functions

def custom_test(custom_search,trs):
    print(custom_search)
    if custom_search == '':
        result = []
    else:
        result = filter_test_record_by_one_name(custom_search, trs)
    select_and_input = interactive(select_test, name=widgets.Dropdown(description="Test:", options=result),trs = fixed(trs),
                      remove_n_start=widgets.Dropdown(description="Remove Cycles (Start of Test):", style={'description_width': 'initial'}, options=[None, 1, 2, 3, 4, 5], value=None),
                      remove_n_end=widgets.Dropdown(description="Remove Cycles (End of Test):", style={'description_width': 'initial'}, options=[None, 1, 2, 3, 4, 5], value=None))
    display(select_and_input)
    return result

def select_test(name, trs, remove_n_start=None, remove_n_end=None):
        if name == None:
            pass
        else:
            test = filter_test_record_by_name(name, trs)[0]

            cycle_traces = ["Discharge Capacity", "Charge Capacity", "Discharge Energy", "Charge Energy", "Coulombic Efficiency", "Total Charge Time", "Total Discharge Time", 
                            "Mean Dis. Potential by Capacity", "Mean Ch. Potential by Capcity"]

            all_cycle_stats = test.get_cycle_stats().rename(columns={"cycle_number": "Cycle Number", "cyc_discharge_capacity": "Discharge Capacity", 'cyc_charge_capacity': "Charge Capacity",
                                                                     'cyc_charge_energy': "Charge Energy", 'cyc_coulombic_efficiency': "Coulombic Efficiency", 
                                                                     'cyc_discharge_energy': "Discharge Energy", 'cyc_total_charge_time': "Total Charge Time", 
                                                                     'cyc_total_discharge_time': "Total Discharge Time", 
                                                                     'cyc_mean_discharge_potential_by_capacity': "Mean Dis. Potential by Capacity", 
                                                                     'cyc_mean_charge_potential_by_capacity': "Mean Ch. Potential by Capcity"})
            # option to remove first n & last cycles
            if remove_n_start or remove_n_end:
                all_cycle_stats = remove_firstn_and_lastn_cycles(all_cycle_stats, remove_n_start, remove_n_end)
            test_anomalies = []
            for trace in cycle_traces:
                # find/remove nan cycles
                nan_cycles, cycle_stats = find_nan_cycles(all_cycle_stats[["Cycle Number", trace]], trace)
                scaled = scale_data(cycle_stats)
                anomalies = run_LOF(scaled, cycle_stats)
                test_anomalies.append(anomalies.merge(nan_cycles, how="outer", sort=True).set_index("Cycle Number")) 

            display(Markdown(f"\n**{name} Anomalies:**"))
            anomalies = pd.concat(test_anomalies, axis=1)
            if anomalies.empty:
                print(f"No Cycle Anomalies Detected for test {test}.\n")
            else:
                display(anomalies.sort_index().fillna("").round(3))
            @widgets.interact(trace=widgets.Dropdown(description="Cycle Trace:", options=cycle_traces))
            def display_trace_data(trace):
    #             print(all_cycle_stats.index)
    #             print(anomalies.index)
                plot_data(all_cycle_stats.set_index("Cycle Number"), anomalies, trace)

            @widgets.interact(cycle=widgets.Combobox(value=None, options=list(all_cycle_stats.index.astype(str)), description='Cycle:', ensure_option=True, continuous_update=False))
            def display_cycle_timeseries(cycle=None):
                if cycle != "":
                    plot_cycle_timeseries(test, int(cycle))

def detect_anomalies(dataset,trs):    
    
    if dataset == 'Custom':
        custom_results = interactive(custom_test,custom_search=widgets.Text(description="Test search phrase:",
                                                                            value='iphone',style={'description_width': 'initial'},continuous_update=False),trs = fixed(trs))
        test_names = custom_results.result
        display(custom_results)
    else:  
        test_names = get_standard_datasets(dataset, trs)
    
        select_and_input = interactive(select_test, name=widgets.Dropdown(description="Test:", options=test_names),trs = fixed(trs),
                      remove_n_start=widgets.Dropdown(description="Remove Cycles (Start of Test):", style={'description_width': 'initial'}, options=[None, 1, 2, 3, 4, 5], value=None),
                      remove_n_end=widgets.Dropdown(description="Remove Cycles (End of Test):", style={'description_width': 'initial'}, options=[None, 1, 2, 3, 4, 5], value=None))
        display(select_and_input)