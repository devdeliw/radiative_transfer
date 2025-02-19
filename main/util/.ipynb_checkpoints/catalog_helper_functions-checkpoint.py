import numpy as np
import pandas as pd

def get_all_filts(t):
    """
    Helper function to get filts in a catalog, regardless of
    masking
    """
    nepochs = len(t['filt'][0])

    # Loop through each time entry and get year
    # from a non-masked source
    filt_arr = []
    for ii in range(nepochs):
        filt_col = t['filt'][:,ii]

        good = np.where(t['x'][:,ii].mask == False)[0]
        if len(good) > 0:
            filt_arr.append(t['filt'][good[0],ii])
        else:
            filt_arr.append(np.nan)
       

    filt_arr = np.array(filt_arr)
   
    return filt_arr

def get_all_dets(t): 
    """
    Helper function to get detectors in a catalog, regardless of
    masking
    """
    nepochs = len(t['det'][0])

    # Loop through each time entry and get year
    # from a non-masked source
    det_arr = []
    for ii in range(nepochs):
        det_col = t['det'][:,ii]

        good = np.where(t['x'][:,ii].mask == False)[0]
        if len(good) > 0:
            det_arr.append(t['det'][good[0],ii])
        else:
            det_arr.append(np.nan)

    det_arr = np.array(det_arr)
   
    return det_arr

def get_matches(t, filt1, det1, filt2, det2): 
    filt = get_all_filts(t)
    det = get_all_dets(t)

    idx1 = np.where( (filt == filt1) & (det == det1) ) 
    idx2 = np.where( (filt == filt2) & (det == det2) ) 

    filt1_str = f"m_vega_ave_{filt1}"
    filt1e_str = f"me_vega_ave_{filt1}"
    filt2_str = f"m_vega_ave_{filt2}" 
    filt2e_str = f"me_vega_ave_{filt2}"

    filt_1 = t['m_vega_avg_F115W'][:,idx1]
    filt_2 = t['m'][:,idx2]

    filt_1_x = t['x'][:, idx1] 
    filt_2_x = t['x'][:, idx2] 
    filt_1_y = t['y'][:, idx1] 
    filt_2_y = t['y'][:, idx2] 

    filt_1me = t['me'][:,idx1]
    filt_2me = t['me'][:,idx2]
    
    good = np.where( (filt_1.mask == False) & (filt_2.mask == False) )
    filt1_x_match = filt_1_x[good]
    filt2_x_match = filt_2_x[good] 
    filt1_y_match = filt_1_y[good] 
    filt2_y_match = filt_2_y[good] 
    m_filt1_match  = filt_1[good]
    m_filt2_match  = filt_2[good]
    me_filt1_match = filt_1me[good]
    me_filt2_match = filt_2me[good]

    filt1_coord = np.vstack((np.array(filt1_x_match), np.array(filt1_y_match)))
    filt2_coord = np.vstack((np.array(filt2_x_match), np.array(filt2_y_match)))

    return filt1_coord, filt2_coord, m_filt1_match, m_filt2_match, me_filt1_match, me_filt2_match

def get_csv_matches(csv_loc, filt1, region1, filt2, region2):
    
    df = pd.read_csv(csv_loc)

    region_map = {
        'NRCB1': '',
        'NRCB2': '.1',
        'NRCB3': '.2',
        'NRCB4': '.3'
    }

    suffix1 = region_map.get(region1, '')
    suffix2 = region_map.get(region2, '')

    column1 = f"{filt1}{suffix1}"  
    column2 = f"{filt2}{suffix2}"      

    array1 = df[column1]
    array2 = df[column2]

    combined_df = pd.DataFrame({column1: array1, column2: array2})
    
    filtered_df = combined_df.dropna()

    return filtered_df[column1], filtered_df[column2]
