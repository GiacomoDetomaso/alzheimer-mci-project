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
        subject_col_name:str
    ) -> pd.DataFrame:
    """
        # TODO doc comments
    """
    # Work on a copy of the DataFrame
    df_copy = df.copy(deep=True)

    # Take the absolute value to remove negative index
    df_copy[time_col_name] = np.abs(df_copy[time_col_name])

    # Change the name of the subjects
    df_copy[subject_col_name] = df_copy[subject_col_name].str.replace('-', '')

    return df_copy[[subject_col_name, time_col_name]]


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
    
    if len(possible_upper_bounds[time_col_name]) != 0:
        upper_bound_idx = possible_upper_bounds[time_col_name].idxmin()

    if len(possible_lower_bounds[time_col_name]) != 0:    
        lower_bound_idx = possible_lower_bounds[time_col_name].idxmax() 

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
        target_col_name:str,
        time_col_name:str='time',
    ) -> pd.DataFrame:
    """
        # TODO doc comments
    """
    subjects_list = source_df[subject_col_name].unique().tolist()
    source_df_copy = source_df.copy(deep=True)

    # Instantiate the output columns filling them with NA values
    source_df_copy[target_col_name] = pd.NA

    for subject in subjects_list:
        # Get sub dataframe related to a particular subject
        sub_target_cond = target_df[subject_col_name] == subject
        sub_source_cond = source_df_copy[subject_col_name] == subject

        args = (target_df[sub_target_cond], time_col_name, target_col_name)

        source_df_copy.loc[sub_source_cond, target_col_name] = (
            source_df_copy
                .loc[sub_source_cond, time_col_name] # Select time column for the slice of target df
                .apply(
                    func=__map_time_to_CDR, # apply the previously defined function
                    args=args
                )
        )

    return source_df_copy[target_col_name]

def align_labels(target_series:pd.Series):
    pass
