#Select the best attributes by applying Mutual Information

def feature_selector_mutual (values_col,target_col,number_features):
    from sklearn.feature_selection import mutual_info_regression
    mic = mutual_info_regression(values_col,target_col,random_state=345)
    print('Los atributos más importantes, acorde con la información mutua son:',
        pd.DataFrame({'Features':list(values_col),
              'Mutual_information':mic}).sort_values(by='Mutual_information',ascending=False)[:number_features])
    features_select=pd.DataFrame({'Features':list(values_col),
              'Mutual_information':mic}).sort_values(by='Mutual_information',ascending=False)['Features'][:number_features].values
    values_reduced = values_col[features_select]
    return(values_reduced,features_select) 
