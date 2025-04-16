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
        cdr_col_name:str,
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
    cdr_ub = possible_upper_bounds.loc[upper_bound_idx, cdr_col_name] if upper_bound_idx != None else None
    cdr_lb = possible_lower_bounds.loc[lower_bound_idx, cdr_col_name] if lower_bound_idx != None else None

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


def get_mapped_target_column(
        target_df:pd.DataFrame, 
        source_df:pd.DataFrame,
        subject_col_name:str,
        target_col_name:str,
        time_col_name:str='time',
    ) -> pd.DataFrame:
    """
        This function returns a column containing the CDR of a subject who underwent 
        an MRI scan in a certain period of time. It maps the time period of the MRI 
        scan with the assessment of CDR made (usually immediately following the scan).
        
        ## Args
            - target_df (pd.DataFrame): it contains the data abount CDR assessment to extract
            - source_df (pd.DataFrame): it contains the data on which the CDR info needs to be added
            - subject_col_name (str): the name of the subject column (the one that contains 
                                      values like OAS30001) in `source_df`
            - target_col_name (str): the name of the column containg target informations in `target_df`
            - diagnosis_col_name (str): the name of the column contining the diagnosis information in target_df`
            - time_col_name (str): the name of the column containig time information inside both df
            
        ## Returns
            A DataFrame with the same rows as `source_df` with the columns 
            [`target_col_name`, `diagnosis_col_name`]
    """
    subjects_list = source_df[subject_col_name].unique().tolist()
    source_df_copy = source_df.copy(deep=True)

    # Instantiate the output columns filling them with NA values
    source_df_copy[target_col_name] = pd.NA

    for subject in subjects_list:
        # Get sub dataframe related to a particular subject
        sub_target_cond = target_df[subject_col_name] == subject
        sub_source_cond = source_df_copy[subject_col_name] == subject

        # Args to pass to pandas apply method: 
        # 1) Target dataframe slice related to actual subject
        # 2) Name of time column
        # 3) Name of CDR column which is the target
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


def simplify_diagnosis_label(
        dx: pd.Series, 
        non_dem:list, 
        ques_dem:list, 
        mapping_dict:dict
    ):
    """
        This function maps the diagnosis labels into two main category: Non-Demented
        and Demented.

        ## Args
            - dx (pd.Series): the series of the diagnosis 
            - non_dem (list): categories to NOT map into dementia
            - mapping_dict (dict): a dictionary that maps labels into numeric values 
                                   in order to make the output ready for other processing functions

        ## Returnes
            The mapped input series
    """
    return (dx.copy(deep=True)
                .map(
                    lambda x: 'Non-Demented' if x in non_dem 
                               else 'Questionable-Demented' if x in ques_dem else 'Demented'
                )
                .map(mapping_dict))            


def get_final_labels(
        df:pd.DataFrame, 
        diagnosis_col_name:str, 
        cdr_col_name:str, 
        mapping_dict:dict,
        reverse_mapping:bool=False
    ) -> pd.Series:
    """
        This function helps retrieving the final versions of the labels: namely
        Non-Demented, Demented and MCI.

        ## Args
            - df (pd.DataFrame): the input DataFrame with the diagnosis column in it
            - diagnosis_col_name (str): the name of the column contining the diagnosis information in `df`
            - cdr_col_name (str): the name of the CDR column inside `df`
            - mapping_dict (dict): a dictionary that maps labels into numeric values 
                                   (the same one passed to `simplify_diagnosis_label`)
            - reverse_mapping (bool): if True returns the labels string instead then mapped labels'number

        ## Returns
            A series containing the dataset labels
    """
    df_copy:pd.DataFrame = df[[diagnosis_col_name, cdr_col_name]].copy(deep=True)

    mapping_function = (
        lambda x: mapping_dict['Questionable-Demented']
                    if x.loc[cdr_col_name] == 0.5 # Define the condition for the MCI labels
                    else x[diagnosis_col_name]
    )

    labels = df_copy.apply(
                func=mapping_function,
                axis='columns'
    )
    
    if reverse_mapping:
        reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}
        
        labels = labels \
                    .map(reverse_mapping_dict) \
                    .map(lambda x: 'MCI' if x == 'Questionable-Demented' else x)

    return labels


# TODO define a function to incorporate all the dataset extraction logic define here