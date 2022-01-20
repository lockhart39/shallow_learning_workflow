def check_for_nulls(data: 'Pandas Dataframe'):
    """Return a dictionary where the keys are fields names from
    a Panda's dataframe and the key's value represents
    how many nulls are present. If the dataframe doesn't have
    any fields with null values it will return a dictionary
    specifying so.
    """
    cols_w_nulls = {}
    null_counter = 0
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            null_counter += 1
            cols_w_nulls[col] = data[col].isnull().sum()
    if null_counter > 0:
        return cols_w_nulls
    else:
        return {'All field values': 'Populated'}
