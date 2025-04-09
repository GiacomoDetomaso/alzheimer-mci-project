import pandas as pd
import numpy as np

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
        target_col_name:str
    ):
    # The possible labels are the ones where the time value is greater or lower than x
    # which is the time value for the current subject in the outer loop. This defines upper 
    # and lower bound to x 
    possible_upper_bounds = sub_target_df.loc[sub_target_df[time_col_name] >= x]
    possible_lower_bounds = sub_target_df.loc[sub_target_df[time_col_name] <= x]

    upper_bound_idx = None
    lower_bound_idx = None
    
    # Calculates the bounds only if the sequence of values is not empty
    if len(possible_upper_bounds[time_col_name]) != 0:
        upper_bound_idx = possible_upper_bounds[time_col_name].idxmin()

    if len(possible_lower_bounds[time_col_name]) != 0:    
        lower_bound_idx = possible_lower_bounds[time_col_name].idxmax() 

    # Extract upper and lower bounds to x
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


def get_CDR_column(
        target_df:pd.DataFrame, 
        source_df:pd.DataFrame,
        subject_col_name:str,
        cdr_col_name:str,
        time_col_name:str='time',
    ) -> pd.Series:
    """
        This function returns a column containing the CDR of a subject who underwent 
        an MRI scan in a certain period of time. It maps the time period of the MRI 
        scan with the assessment of CDR made (usually immediately following the scan).
        
        ## Args
            - target_df (pd.DataFrame): it contains the data abount CDR assessment to extract
            - source_df (pd.DataFrame): it contains the data on which the CDR info needs to be added
            - subject_col_name (str): the name of the subject column (the one that contains 
                                      values like OAS30001) in `source_df`
            - cdr_col_name (str): the name of the column containg cdr informations in `target_df`
            - time_col_name (str): the name of the column containig time information inside both df
            
        ## Returns
            A series with the same number of rows as `source_df`, with the cdr to add to it
    """
    subjects_list = source_df[subject_col_name].unique().tolist()
    source_df_copy = source_df.copy(deep=True)

    # Instantiate the output columns filling them with NA values
    source_df_copy[cdr_col_name] = pd.NA

    for subject in subjects_list:
        # Get sub dataframe related to a particular subject
        sub_target_cond = target_df[subject_col_name] == subject
        sub_source_cond = source_df_copy[subject_col_name] == subject

        # Args to pass to pandas apply method: 
        # 1) Target dataframe slice related to actual subject
        # 2) Name of time column
        # 3) Name of CDR column which is the target
        args = (target_df[sub_target_cond], time_col_name, cdr_col_name)

        source_df_copy.loc[sub_source_cond, cdr_col_name] = (
            source_df_copy
                .loc[sub_source_cond, time_col_name] # Select time column for the slice of target df
                .apply(
                    func=__map_time_to_CDR, # apply the previously defined function
                    args=args
                )
        )

    return source_df_copy[cdr_col_name]


def align_labels(df:pd.DataFrame, subject_col_name:str, cdr_col_name:str):
    subjects_list = df[subject_col_name].unique().tolist()
    
    for subject in subjects_list:
        sub_df_cond = df[subject_col_name] == subject

        if not df.loc[sub_df_cond, cdr_col_name].is_monotonic_increasing:
            pass
    
