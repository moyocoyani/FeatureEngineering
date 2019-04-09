class TransformationCatalogue:
    
    def __init__(self,dataset,feature_1,feature_2='None'):
        if feature_2 != 'None':
            self.dataset = dataset.copy()
            self.feature_1 = self.dataset[feature_1].values
            self.name1 = feature_1
            self.feature_2 = self.dataset[feature_2].values
            self.name2 = feature_2
        else:
            self.dataset = dataset.copy()
            self.feature_1 = self.dataset[feature_1].values
            self.name1 = feature_1
            self.name2 = feature_2
            
    def positive_test(self,feature):
        if self.dataset[feature].min() > 0:
            return 1
        else: return 0

    def cont_to_freq(self,feature):
        temp_df=pd.DataFrame({'Total':self.dataset.groupby(feature).size()})
        temp_df['Total']=temp_df['Total']/len(temp_df)
        return temp_df['Total'].values    
        
    def log_transform(self):
        #Validando si es mayor a cero
        if self.positive_test(self.name1) == 1:
            var_name = 'log'+self.name1
            self.dataset[var_name] = np.log(self.feature_1)
            return self.dataset
        else:
            print('valores menores a cero, imposible transformar')
            
    def exp_transform(self):
        var_name = 'exp'+self.name1
        self.dataset[var_name] = np.exp(self.feature_1)
        return self.dataset
    
    def quadratic_transform(self):
        var_name = 'quad'+self.name1
        self.dataset[var_name] = (self.feature_1)*(self.feature_1)
        return self.dataset
    
    def cubic_transform(self):
        var_name = 'cubic'+self.name1
        self.dataset[var_name] = (self.feature_1)*(self.feature_1)*(self.feature_1)
        return self.dataset
    
    def sin_transform(self):
        var_name = 'sin'+self.name1
        self.dataset[var_name] = np.sin(self.feature_1)
        return self.dataset
    
    def cosin_transform(self):
        var_name = 'cosin'+self.name1
        self.dataset[var_name] = np.cos(self.feature_1)
        return self.dataset
    
    def cosin_transform(self):
        var_name = 'cosin'+self.name1
        self.dataset[var_name] = np.cos(self.feature_1)
        return self.dataset
    
    def dev_transform(self):
        var_name = 'dev'+self.name1
        mean_ = np.mean(self.feature_1)
        self.dataset[var_name] = self.feature_1 - mean_
        return self.dataset
    
    def shannon_transform(self):
        var_name = 'shannon' + self.name1
        p1 = self.cont_to_freq(self.name1)
        self.dataset[var_name]= -p1 *np.log(p1)
        return self.dataset
    
    def multiply_tr(self):
        var_name = 'prod' + self.name1 +'_' + self.name2
        if self.name2 != 'None':
            self.dataset[var_name]= self.feature_1*self.feature_2
        else:
            print('No hay atributo 2')
        return self.dataset
    
    def divide_tr(self):
        #Validando si es mayor a cero
        if (self.positive_test(self.name1) == 1) and (self.name2 != 'None'):
            var_name = 'division' + self.name2 +'_' + self.name1
            self.dataset[var_name] = self.feature_2/self.feature_1
            return self.dataset
        else:
            print('valores menores a cero, imposible transformar; o solo existe un atributo')
            
