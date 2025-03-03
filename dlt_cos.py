import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Merge RF and BLE raw data files and returns merged dataframe
def merge_files(rffile_path, blefile_path):
    merge_col = 'deviceName'

    try:
        df1 = pd.read_csv(rffile_path, engine='python')
        df1 = df1.dropna(how = 'all')
        cell_id = df1.iloc[1,2]
        df1 = df1.drop(['id', 'ch37_RX_Sens', 'ch38_RX_Sens', 'ch39_RX_Sens', 'ch37_RX_PER', 'ch38_RX_PER', 'ch39_RX_PER'], axis = 1)      

        df2 = pd.read_csv(blefile_path, engine='python')
        df2 = df2.dropna(how = 'all')
        df2 = df2.drop(['id', 'created_at', 'toolNumber', 'lotIdent', 'buildUUID', 'CMAC', 'passkey', 
                        'W2Accuracy10nA', 'W2Accuracy100nA', 'W2Accuracy2500nA', 'W2Accuracy300nA', 
                        'W3Accuracy10nA', 'W3Accuracy100nA', 'W3Accuracy2500nA', 'W3Accuracy300nA', 
                        'W4Accuracy10nA', 'W4Accuracy100nA', 'W4Accuracy2500nA', 'W4Accuracy300nA', 'spare1', 'spare2'], axis = 1)

        merged_df = pd.merge(df1, df2, on = merge_col, how = "left")
        merged_df['VBattDelta'] = merged_df['VBattUnloaded_afe'] - merged_df['VBattLoaded_afe']

        merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
        return merged_df, cell_id

    except FileNotFoundError:
         print("Error: One or both input files not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Add acceptance criteria to data columns and returns dataframe with PASS(0)/FAIL(1) per criteria for all device
def add_AC(merged_df):
    tx_low = -3
    tx_high = 9
    txmod_low = -97040
    txmod_high = 97040
    refv_low = 820
    refv_high = 880
    w1v_low = 1355
    w1v_high = 1415
    w1l_low = -100
    w1l_high = 100
    w1a10_low = -200
    w1al0_high = 200
    w1a100_low = -120
    w1a100_high = 120
    vbattun_low = 3040.0
    vbattun_high = 3366.0
    vbattdelta_low = 30
    vbattdelta_high = 240
    crystal_low = 3275817
    crystal_high = 3277783
    
    no_BLE_rows = merged_df[merged_df['referenceVoltage'].isna()].filter(['deviceName', 'created_at'])
    no_BLE_rows['No_BLE'] = 1
    merged_df = merged_df[merged_df['referenceVoltage'].notna()]
    
    filtered_df = merged_df.filter(['deviceName', 'created_at', 
                                    'ch37_TX_Power', 'ch38_TX_Power', 'ch39_TX_Power',
                                    'ch37_TX_MOD', 'ch38_TX_MOD', 'ch39_TX_MOD',
                                    'ch37_RX_SPOT', 'ch38_RX_SPOT', 'ch39_RX_SPOT',
                                    'W1Leakage', 'W1Accuracy10nA', 'W1Accuracy100nA',
                                    'VBattUnloaded_afe', 'VBattDelta', 'crystalFreq'], axis = 1)
    filtered_df['ch37_RX_SPOT'] = filtered_df['ch37_RX_SPOT'].apply(lambda x: 0 if ((x == 'PASS')) else 1)
    filtered_df['ch38_RX_SPOT'] = filtered_df['ch38_RX_SPOT'].apply(lambda x: 0 if ((x == 'PASS')) else 1)
    filtered_df['ch39_RX_SPOT'] = filtered_df['ch39_RX_SPOT'].apply(lambda x: 0 if ((x == 'PASS')) else 1)

    filtered_df['RX_SPOT_Sum'] = filtered_df['ch37_RX_SPOT'] + filtered_df['ch38_RX_SPOT'] + filtered_df['ch39_RX_SPOT']
    filtered_df['RX_SPOT_Sing_CH'] = filtered_df['RX_SPOT_Sum'].apply(lambda x: 1 if (x == 1) else 0)
    filtered_df['RX_SPOT_Multi_CH'] = filtered_df['RX_SPOT_Sum'].apply(lambda x: 1 if (x > 1) else 0)

    filtered_df['ch37_TX_Power'] = filtered_df['ch37_TX_Power'].apply(lambda x: 0 if ((x >= tx_low) & (x <= tx_high)) else 1)
    filtered_df['ch38_TX_Power'] = filtered_df['ch38_TX_Power'].apply(lambda x: 0 if ((x >= tx_low) & (x <= tx_high)) else 1)
    filtered_df['ch39_TX_Power'] = filtered_df['ch39_TX_Power'].apply(lambda x: 0 if ((x >= tx_low) & (x <= tx_high)) else 1)

    filtered_df['TX_Power_Sum'] = filtered_df['ch37_TX_Power'] + filtered_df['ch38_TX_Power'] + filtered_df['ch39_TX_Power']
    filtered_df['TX_Power_Sing_CH'] = filtered_df['TX_Power_Sum'].apply(lambda x: 1 if (x == 1) else 0)
    filtered_df['TX_Power_Multi_CH'] = filtered_df['TX_Power_Sum'].apply(lambda x: 1 if (x > 1) else 0)

    filtered_df['ch37_TX_MOD'] = filtered_df['ch37_TX_MOD'].apply(lambda x: 0 if ((x >= txmod_low) & (x <= txmod_high)) else 1)
    filtered_df['ch38_TX_MOD'] = filtered_df['ch38_TX_MOD'].apply(lambda x: 0 if ((x >= txmod_low) & (x <= txmod_high)) else 1)
    filtered_df['ch39_TX_MOD'] = filtered_df['ch39_TX_MOD'].apply(lambda x: 0 if ((x >= txmod_low) & (x <= txmod_high)) else 1)

    filtered_df['TX_MOD_Sum'] = filtered_df['ch37_TX_MOD'] + filtered_df['ch38_TX_MOD'] + filtered_df['ch39_TX_MOD']
    filtered_df['TX_MOD_Sing_CH'] = filtered_df['TX_MOD_Sum'].apply(lambda x: 1 if (x == 1) else 0)
    filtered_df['TX_MOD_Multi_CH'] = filtered_df['TX_MOD_Sum'].apply(lambda x: 1 if (x > 1) else 0)

    filtered_df['W1Leakage'] = filtered_df['W1Leakage'].apply(lambda x: 0 if ((x >= w1l_low) & (x <= w1l_high)) else 5)
    filtered_df['W1Accuracy10nA'] = filtered_df['W1Accuracy10nA'].apply(lambda x: 0 if ((x >= w1a10_low) & (x <= w1al0_high)) else 1)
    filtered_df['W1Accuracy100nA'] = filtered_df['W1Accuracy100nA'].apply(lambda x: 0 if ((x >= w1a100_low) & (x <= w1a100_high)) else 3)

    filtered_df['Leak_Acc_Sum'] = filtered_df['W1Leakage'] + filtered_df['W1Accuracy10nA'] + filtered_df['W1Accuracy100nA']
    filtered_df['W1Leakage'] = filtered_df['Leak_Acc_Sum'].apply(lambda x: 1 if (x == 5) else 0)
    filtered_df['W1Accuracy10nA'] = filtered_df['Leak_Acc_Sum'].apply(lambda x: 1 if (x == 1) else 0)
    filtered_df['W1Accuracy100nA'] = filtered_df['Leak_Acc_Sum'].apply(lambda x: 1 if (x == 3) else 0)
    filtered_df['W1Leak_W1Acc'] = filtered_df['Leak_Acc_Sum'].apply(lambda x: 1 if (x > 5) else 0)

    filtered_df['VBattUnloaded_afe'] = filtered_df['VBattUnloaded_afe'].apply(lambda x: 0 if ((x >= vbattun_low) & (x <= vbattun_high)) else 1)
    filtered_df['VBattDelta'] = filtered_df['VBattDelta'].apply(lambda x: 0 if ((x >= vbattdelta_low) & (x <= vbattdelta_high)) else 2)

    filtered_df['VBatt_Sum'] = filtered_df['VBattUnloaded_afe'] + filtered_df['VBattDelta']
    filtered_df['VBattUnloaded_VBattDelta'] = filtered_df['VBatt_Sum'].apply(lambda x: 1 if (x == 3) else 0)
    filtered_df['VBattUnloaded_afe'] = filtered_df['VBatt_Sum'].apply(lambda x: 1 if (x == 1) else 0)
    filtered_df['VBattDelta'] = filtered_df['VBatt_Sum'].apply(lambda x: 1 if (x == 2) else 0)

    filtered_df['crystalFreq'] = filtered_df['crystalFreq'].apply(lambda x: 0 if ((x >= crystal_low) & (x <= crystal_high)) else 1)

    filtered_df = pd.concat([filtered_df, no_BLE_rows], ignore_index = True)
    filtered_df = filtered_df.fillna(0)

    return(filtered_df)

# Split dataframe into multiple dataframe based on one-week timeframe and return dataframe with count for all failure modes
def cnt_By_Week(filtered_df):
    filtered_df = filtered_df.groupby([pd.Grouper(key='created_at', freq='W')]).agg({'deviceName': 'count', 'RX_SPOT_Sing_CH': 'sum', 'RX_SPOT_Multi_CH': 'sum',
                                                                                     'TX_MOD_Sing_CH': 'sum', 'TX_MOD_Multi_CH': 'sum', 'TX_Power_Sing_CH': 'sum', 'TX_Power_Multi_CH': 'sum',
                                                           'W1Leakage': 'sum', 'W1Accuracy10nA': 'sum', 'W1Accuracy100nA': 'sum', 'W1Leak_W1Acc': 'sum',
                                                           'VBattUnloaded_VBattDelta': 'sum', 'VBattUnloaded_afe': 'sum', 'VBattDelta': 'sum',
                                                           'crystalFreq': 'sum', 'No_BLE': 'sum'}).reset_index()
    return(filtered_df)

# Plot Pareto Chart
def plot_Pareto(filtered_df, cell_id):

    weeks_lbl = filtered_df['created_at'].dt.date.to_list()

    fail_cnt = {'W1Leakage': filtered_df['W1Leakage'], 'RX_SPOT': filtered_df['RX_SPOT_Multi_CH'], 
                'VBattUnloaded_VBattDelta': filtered_df['VBattUnloaded_VBattDelta'], 'TX_PWR': filtered_df['TX_Power_Multi_CH']}
   
    x = np.arange(len(weeks_lbl))
    width = 0.1
    multiplier = 0

    fig, ax1 = plt.subplots(1, layout='constrained')

    for attribute, measurement in fail_cnt.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, measurement, width, label = attribute)
        ax1.bar_label(rects, padding = 1)
        multiplier += 1
    
    ax1.set_ylabel('Count')
    ax1.set_title(cell_id)
    ax1.set_xticks(x + width, weeks_lbl)
    ax1.legend(loc = 'upper left', ncols = 4)

    plt.show()


rffile_path = r"C:\Users\kotobf2\OneDrive - Medtronic PLC\Desktop\Cel02-RF-2.csv"
blefile_path = r"C:\Users\kotobf2\OneDrive - Medtronic PLC\Desktop\Cel02-BLE-2.csv"


[mergeddf, cellid] = merge_files(rffile_path, blefile_path)

mergeddf = add_AC(mergeddf)
cleandf = cnt_By_Week(mergeddf)
print(cleandf)
plot_Pareto(cleandf, cellid)
