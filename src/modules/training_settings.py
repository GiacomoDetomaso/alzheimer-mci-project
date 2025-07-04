import pandas as pd


numeric_cols = [
        'TOTAL_HIPPOCAMPUS_VOLUME', 
        'Left-Hippocampus_volume', 
        'lh_parahippocampal_thickness',
        'lh_parahippocampal_volume',
        'rh_parahippocampal_volume',
        'rh_parahippocampal_thickness',
        'Right-Hippocampus_volume'
]

def set_numeric_cols(numeric_cols):
    numeric_cols = numeric_cols


def __remove_instances_by_label(train_df, val_df, test_df, value):
    train_df = train_df.drop(index=train_df[train_df['label'] == value].index)
    val_df = val_df.drop(index=val_df[val_df['label'] == value].index)
    test_df = test_df.drop(index=test_df[test_df['label'] == value].index)

    return train_df, val_df, test_df


def set_classification_problem_dataframes(
        train_df, 
        val_df, 
        test_df, 
        type='binary', 
        subtype='CNvsAD'
    ):
    if type == 'binary' and subtype == 'all':
        train_df = train_df.drop(index=train_df[train_df['augmentation'] != 'normal'].index)
        val_df = val_df.drop(index=val_df[val_df['augmentation'] != 'normal'].index)

    drop_cols = ['Subject', 'CDRTOT', 'Unnamed: 0', 'MR_session_original', 'augmentation']

    train_df = train_df.drop(columns=drop_cols).sort_values(by='MR_session').reset_index(drop=True)
    val_df = val_df.drop(columns=drop_cols[:3]).sort_values(by='MR_session').reset_index(drop=True)
    test_df = test_df.drop(columns=drop_cols[:3]).sort_values(by='MR_session').reset_index(drop=True)

    if type == 'binary' and subtype == 'CNvsAD':
        train_df, val_df, test_df = __remove_instances_by_label(train_df, val_df, test_df, 'Early-stage')
        label_mapping = {'Cognitevely-normal': 0, 'Demented':1}
    elif type == 'binary' and subtype == 'CNvsEarly':
        train_df, val_df, test_df = __remove_instances_by_label(train_df, val_df, test_df, 'Demented')
        label_mapping = {'Cognitevely-normal': 0, 'Early-stage':1}
    elif type == 'binary' and subtype == 'ADvsEarly':
        train_df, val_df, test_df = __remove_instances_by_label(train_df, val_df, test_df, 'Cognitevely-normal')
        label_mapping = {'Early-stage': 0, 'Demented':1}

    if type == 'ternary':
        label_mapping = {'Cognitevely-normal': 0,'Early-stage': 1, 'Demented': 2}

    train_df['label'] = train_df['label'].map(label_mapping)
    val_df['label'] = val_df['label'].map(label_mapping)
    test_df['label'] = test_df['label'].map(label_mapping)

    return train_df, val_df, test_df


def __get_dataset(df:pd.DataFrame):

    dataset = []

    for _, df_row in df.iterrows():
        dict_row = {}
        dict_row['left'] = df_row['left']
        dict_row['right'] = df_row['right']
        dict_row['data'] = df_row[numeric_cols].astype(float).to_numpy()
        dict_row['experiment'] = df_row['MR_session']
        dict_row['label'] = df_row['label']

        dataset.append(dict_row)

    return dataset


def __get_path(session, direction, directory):
    file = 'hippocampus_left_processed.nii' if direction == 'left' else 'hippocampus_right_processed.nii'
    
    return f'{directory}/{session}/{file}'


def get_datasets_dict(
    df, 
    base_dir, 
):
    df['left'] = df['MR_session'].map(lambda x: __get_path(x, 'left', base_dir))
    df['right'] = df['MR_session'].map(lambda x: __get_path(x, 'right', base_dir))

    return __get_dataset(df)
