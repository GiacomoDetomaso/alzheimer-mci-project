import pandas as pd
import numpy as np

"""
    A module which provides functions to manipulate two dataframe: one related 
    to the freesurfers scan and the other one related to diagnosis. The purpose
    of this module is to provide high level APIs to the "2_target_label_definition.ipynb" 
    notebook that extracts a new dataset with the final labels (from the aforementioned datasets).
"""

def get_time_column(col:pd.Series) -> pd.Series:
    """
        This function extract the time period of the vist 
        from each input series of OASIS subjects

        ## Args
            - col (pd.Series): the input series which contains the OASIS
            label related to an experiment (e.g. OAS30001_MR_d0129/OAS30001_MR_d0129)

        ## Returns
            - A series that contains the time values corresponding to the input series      
    """
    # The time information is located at the second element
    # of the split method. Ignore the first charachter of the 
    # resulting string to treat the resulting series as integer
    return col.map(lambda s: s.split('_')[2][1:]).astype(int)


def fix_negative_time_label(
        df:pd.DataFrame, 
        time_col_name:str, 
        subject_session_col_name:str
    ) -> pd.DataFrame:
    """
        This function is used to fix some errors on the OASIS dataset involving 
        subjects with negative day to visit value.
        
        ## Args
            df (pd.DataFrame): input dataframe
            time_col_name (str): the name of the column containig time information inside `df`
            subject_session_col_name (str): the name of the column containg the subject 
                                            session label in `df`
        ## Returns
            An updated `pd.DataFrame` formatted in the following way: `df[[subject_col_name, time_col_name]]`
    """
    # Work on a copy of the DataFrame
    df_copy = df.copy(deep=True)

    # Take the absolute value to remove negative index
    df_copy[time_col_name] = np.abs(df_copy[time_col_name])

    # Change the name of the subjects
    df_copy[subject_session_col_name] = df_copy[subject_session_col_name].str.replace('-', '')

    return df_copy[[subject_session_col_name, time_col_name]]


def __map_time_to_CDR(
        x:int, 
        sub_target_df:pd.DataFrame, 
        time_col_name:str, 
        target_col_name:str,
    ):
    # The possible labels are the ones where the time value in the target_df is greater or lower 
    # than x (which is the time value for the current subject in the outer loop). 
    # This defines upper and lower bound to x, based on diagnosis' time in sub_target_df
    possible_upper_bounds = sub_target_df.loc[sub_target_df[time_col_name] >= x]
    possible_lower_bounds = sub_target_df.loc[sub_target_df[time_col_name] <= x]

    upper_bound_idx = None
    lower_bound_idx = None
    
    # Calculate the upper bound only if the sequence of values is not empty
    if len(possible_upper_bounds[time_col_name]) != 0:
        upper_bound_idx = possible_upper_bounds[time_col_name].idxmin()

    # Calculate lower bound only if the the sequence of values is not empty
    if len(possible_lower_bounds[time_col_name]) != 0:    
        lower_bound_idx = possible_lower_bounds[time_col_name].idxmax() 

    # Extract upper and lower bounds corresponding cdr value
    cdr_ub = possible_upper_bounds.loc[upper_bound_idx, target_col_name] if upper_bound_idx != None else None
    cdr_lb = possible_lower_bounds.loc[lower_bound_idx, target_col_name] if lower_bound_idx != None else None

    if cdr_ub == None:
        # The upper bound cdr is empty
        final_cdr = cdr_lb
    elif cdr_lb == None:
        # The lower bound cdr is empty
        final_cdr = cdr_ub
    elif (cdr_lb is not None) and (cdr_ub is not None) and (cdr_lb == cdr_ub):
        # If both cdr are equal there is no need to find the closest one to when x
        final_cdr = cdr_lb
    else:
        # Default case: select the cdr temporally closest to x
        lb_distance_x = abs(possible_lower_bounds.loc[lower_bound_idx, time_col_name] - x)
        ub_distance_x = abs(possible_upper_bounds.loc[upper_bound_idx, time_col_name] - x)
        
        final_cdr = cdr_ub if ub_distance_x < lb_distance_x else cdr_lb

    return final_cdr


def put_first(df, firsts_columns):
    df_copy = df.copy(deep=True)
    columns = df_copy.columns.to_list()

    for col in firsts_columns:
        columns.remove(col)

    return df_copy.reindex(columns=firsts_columns + columns)


def session_matchup(df_left, df_right, upper_bound, lower_bound):
    # Create a Day column from ID
    # Use the "dXXXX" value from the ID/label in the first column
    # pandas extract will pull that based on a regular expression no matter where it is.

    df1 = df_left.copy(deep=True)
    df2 = df_right.copy(deep=True)

    df1['Day'] = (df1
                    .iloc[:, 0]
                    .str
                    .extract(r'(d\d{4})', expand=False)
                    .str
                    .strip()
                    .apply(lambda x: int(x.split('d')[1]))
                )
    
    df2['Day'] = (df2
                    .iloc[:,0]
                    .str
                    .extract(r'(d\d{4})', expand=False)
                    .str
                    .strip()
                    .apply(lambda x: int(x.split('d')[1]))
                )
    
    if df2.columns[1] != 'Subject':
        df2 = df2.rename(columns={df2.columns[1]: 'Subject'})

    for index, row in df2.iterrows():
        c1 = (df1['Subject'] == row['Subject'])
        c2 = (df1['Day'] < row['Day'] + upper_bound)
        c3 = (df1['Day'] > row['Day'] - lower_bound)

        mask = c1 & c2 & c3   
        
        for name in row.index:
            df1.loc[mask, name] = row[name]

    # Drop rows of which a match was not found
    df1.dropna(subset=[df2.columns.values[0]], inplace=True)

    return df1


def align_labels(df:pd.DataFrame, subject_col_name:str, target_col_name:str) -> pd.Series:
    """
        This function fix dataset errors where there are some cdr for certain patients
        whose sequence is not monotonically increasing. For example the following wrong 
        sequence [0 .0.5 0 0.5 0] becomes [0 0.5 0.5 0.5 0.5].

        ## Args

        - df (pd.DataFrame): DataFrame on which the operations will be applied (actually the DF is deep copied)
        - subject_col_name (str): the name of the subject column (the one that contains 
                                      values like OAS30001) in `df`
        - target_col_name (str): the name of the column to align in `df`. It must be numeric.

        ## Returns
            A series with the same number of rows as `df`, with the cdr aligned 
    """
    subjects_list = df[subject_col_name].unique().tolist()
    df_copy = df.copy(deep=True)
    
    for subject in subjects_list:
        sub_df_cond = df_copy[subject_col_name] == subject

        target_series = df_copy.loc[sub_df_cond, target_col_name]

        if not target_series.is_monotonic_increasing:
            target_list = target_series.tolist()
            print(subject)
            
            for i in range(1, len(target_list), 1):
                # Update the cdr by substituting element i with element i-1 when element i 
                # is less then element i-1, since cdr must be monotonically increasing
                target_list[i] = target_list[i] if (target_list[i-1] <= target_list[i]) else target_list[i-1]
                
            # Assign the updated series on the current slice of the dataframe
            df_copy.loc[sub_df_cond, target_col_name] = pd.Series(target_list, target_series.index)

    return df_copy[target_col_name]