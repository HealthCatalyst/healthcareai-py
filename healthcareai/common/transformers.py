"""
Transformers

This module contains transformers for preprocessing data. Most operate on DataFrames and are named appropriately.

"""
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from healthcareai.common.healthcareai_error import HealthcareAIError

SUPPORTED_IMPUTE_STRATEGY = ['MeanMode', 'RandomForest']


class DataFrameImputer( TransformerMixin ):
    
    """
    Impute missing values in a dataframe.
    
    Parameters
    ----------
    impute : boolean, default=True
    	If True, imputation of missing value takes place.
    	If False, imputation of missing value doesn't happens.
    	
    verbose : boolean, default=True
    	Controls the verbosity.
    	If False : No text information will be shown about imputation of missing values
    	
    imputeStrategy : string, default='MeanMode'
    	It decides the technique to be used for imputation of missing values.
    	If imputeStrategy = 'MeanMode', Columns of dtype object or category 
        (assumed categorical) and imputed by the mode value of that column. 
        Columns of other types (assumed continuous) : by mean of column.
    	
        If imputeStrategy = 'RandomForest', Columns of dtype object or category 
         (assumed categorical) : imputed using RandomForestClassifier. 
         Columns of other types (assumed continuous) : imputed using RandomForestRegressor
    			
    
    tunedRandomForest : boolean, default=False
    	If set to True, RandomForestClassifier/RandomForestRegressor to be used for 
    	imputation of missing values are tuned using grid search and K-fold cross 
    	validation.
    	
    	Note:
    	If set to True, imputation process may take longer time depending upon size of 
    	dataframe and number of columns having missing values.
    	
    numeric_columns_as_categorical : List of type String, default=None
    	List of column names which are numeric(int/float) in dataframe, but by nature 
    	they are to be considered as categorical.
    	
    	For example:
    	There is a column JobCode( Levels : 1,2,3,4,5,6)
    	If there are missing values in JobCode column, panadas will by default convert 
    	this column into type float.
    	
    	
    	If numeric_columns_as_categorical=None
    	Missing values of this column will be imputed by Mean value of JobCode column.
    	type of 'JobCode' column will remain float. 
    	
        If numeric_columns_as_categorical=['JobCode']
        Missing values of this column will be imputed by mode value of JobCode column.
        Also final type of 'JobCode' column will be numpy.object 
						 			
    """
    def __init__(self, impute=True, verbose=True, imputeStrategy='MeanMode', tunedRandomForest=False, 
                 numeric_columns_as_categorical=None ):
        self.impute = impute
        self.object_columns = None
        self.fill = None
        self.verbose = verbose
        
        self.impute_Object = None
        self.imputeStrategy = imputeStrategy
        self.tunedRandomForest = tunedRandomForest
        self.numeric_columns_as_categorical = numeric_columns_as_categorical
        

    def fit(self, X, y=None):
        """
        Description:
        ------------
        
        If imputeStrategy is : 'MeanMode' / None
            Missing value to be imputed are calculated using Mean and Mode of corresponding columns.
            1. Columns specified in 'numeric_columns_as_categorical' are explicitly converted into dtype='object'
            2. Values to be imputed are calculated and stored in variable: self.fill 
            3. Later inside transform function, the same values will be filled in place of missing values.
        
        If imputeStrategy is : 'RandomForest'
            1. Class object of DataFrameImputerRandomForest is created
            2. fit function of DataFrameImputerRandomForest class is called.
        """
        
        if self.impute is False:
            return self
        
            
        if ( self.imputeStrategy=='MeanMode' or self.imputeStrategy==None ):
            
            if( self.numeric_columns_as_categorical is not None ):
                for col in self.numeric_columns_as_categorical:
                    if( col not in list(X.columns) ):
                        raise HealthcareAIError('Column = {} mentioned in numeric_columns_as_categorical is not present in dataframe'.format(col))
                    else:
                        X[col] = X[col].astype( dtype='object', copy=True, error='raise' )
    
            # Grab list of object column names before doing imputation
            self.object_columns = X.select_dtypes(include=['object']).columns.values
            
            num_nans = X.isnull().sum().sum()
            num_total = X.shape[0] * X.shape[1]
            percentage_imputed = num_nans / num_total * 100

            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O')
                                      or pd.api.types.is_categorical_dtype(X[c])
                                   else X[c].mean() for c in X], index=X.columns)

            if self.verbose:
                print("Percentage Imputed: %.2f%%" % percentage_imputed)
                print("Note: Impute will always happen on prediction dataframe, otherwise rows are dropped, and will lead "
                      "to missing predictions")

            # return self for scikit compatibility
            return self
        elif ( self.imputeStrategy=='RandomForest' ):
            self.impute_Object = DataFrameImputerRandomForest( tunedRandomForest=self.tunedRandomForest, 
                                                              numeric_columns_as_categorical=self.numeric_columns_as_categorical, 
                                                              impute=self.impute, verbose=self.verbose )
            self.impute_Object.fit(X)
            return self
        else:
            raise HealthcareAIError('A imputeStrategy must be one of these types: {}'.format(SUPPORTED_IMPUTE_STRATEGY))

            

    def transform(self, X, y=None):
        """
        Description:
        ------------
        
        If imputeStrategy is : 'MeanMode' / None
            Missing value to be imputed are calculated using Mean and Mode of corresponding columns.
            1. Missing values of dataframe ae filled using self.fill variable(generated in fill() function )
            2. Columns specified in 'numeric_columns_as_categorical' are explicitly converted into dtype='object'
            3. Columns captured in 'self.object_columns' during fill() function are ensured to be of dtype='object'
            
        If imputeStrategy is : 'RandomForest'
            1. Already Class object of DataFrameImputerRandomForest is created during fill() function.
            2.. Now transform() function of DataFrameImputerRandomForest class is called.
        """
        
        # Return if not imputing
        if self.impute is False:
            return X
        
        if ( self.imputeStrategy=='MeanMode' or self.imputeStrategy==None ):
            result = X.fillna(self.fill)
            
            if( self.numeric_columns_as_categorical is not None ):
                for col in self.numeric_columns_as_categorical:
                    result[col] = result[col].astype( dtype='object', copy=True, error='raise' )

            for i in self.object_columns:
                if result[i].dtype not in ['object', 'category']:
                    result[i] = result[i].astype('object')

            return result
        elif ( self.imputeStrategy=='RandomForest' ):
            result = self.impute_Object.transform(X)
            return result
        else:
            raise HealthcareAIError('A imputeStrategy must be one of these types: {}'.format(SUPPORTED_IMPUTE_STRATEGY))




class DataFrameImputerRandomForest( TransformerMixin ):

    """
    Impute missing values in a dataframe using RandomForest models.
    
    Parameters
    ----------
    impute : boolean, default=True
    	If True, imputation of missing value takes place.
    	If False, imputation of missing value doesn't happens.
    	
    verbose : boolean, default=True
    	Controls the verbosity.
    	If False : No text information will be shown about imputation of missing values
    
    tunedRandomForest : boolean, default=False
    	If set to True, RandomForestClassifier/RandomForestRegressor to be used for 
    	imputation of missing values are tuned using grid search and K-fold cross 
    	validation.
    	
    	Note:
    	If set to True, imputation process may take longer time depending upon size of 
    	dataframe and number of columns having missing values.
    	
    numeric_columns_as_categorical : List of type String, default=None
    	List of column names which are numeric(int/float) in dataframe, but by nature 
    	they are to be considered as categorical.
    	
    	For example:
    	There is a column JobCode( Levels : 1,2,3,4,5,6)
    	If there are missing values in JobCode column, panadas will by default convert 
    	this column into type float.
    	
    	
    	If numeric_columns_as_categorical=None
    	Missing values of this column will be imputed by Mean value of JobCode column.
    	type of 'JobCode' column will remain float. 
    	
        If numeric_columns_as_categorical=['JobCode']
        Missing values of this column will be imputed by mode value of JobCode column.
        Also final type of 'JobCode' column will be numpy.object 
						 			
    """
    

    def __init__(self, impute=True, verbose=True, tunedRandomForest=False, numeric_columns_as_categorical=None ):
        self.impute = impute
        self.object_columns = None
        self.fill = None
        self.verbose = verbose
        
        self.tunedRandomForest = tunedRandomForest
        self.numeric_columns_as_categorical=numeric_columns_as_categorical
        self.fill_dict = {}

    def fit(self, X, y=None):
        """
        Description:
            1. It segregate the list of all columns into 3 parts:
                1. cat_list = list of categorical columns
                2. num_list = list of numeric columns
                3. num_as_cat_list = numeric columns to be considered as categorical (provided by user)
            2. First of all missing values of num_as_cat_list are filled using 'Mode' values 
                by calling function : getNumericAsCategoricalImputedData.
                At this point calculation of missing vales of [num_as_cat_list] columns is completed.
            3. New dataframe is constructed as : 
                X_NumericAsCategoricalImputed(now don't have any null values) + X[ cat_list+num_list ](still have null values)
                At this point calculation of missing vales of [num_as_cat_list + num_list columns] is completed.
            4. Then after missing values of Numreic colums are imuted by calling function : getNumericImputedData()
            5. New dataframe is constructed as : 
                NumericImputedData(now dont have any missing values) + X[ cat_list ](still have missing values)                                                                                                          missing values)
            6. Now missing values of Categorical colums are imuted by calling function : getCategoricalImputedData()
            7. At this point calculation of missing vales of all columns is completed.
            8. While imputing any columns, corresponding entry is made in fill_dict as:
                key   : column name
                value : array of predicted values for missing cells
                ***This dictionary will be used in transform function to impute the missing values.        
        
        Local Variables:
            main_df (pd.Dataframe)  = copy of original dataframe(having missing values)
            X_backup (pd.Dataframe) = It will be used as backup of dataframe: X
            cat_list (List of Strings) = List of categorical columns
            num_list (List of Strings) = List of numeric columns
            num_as_cat_list = numeric columns tobe considered as categorical (provided by user)
            X_NumericAsCategoricalImputed(pd.Dataframe) =  dataframe having only num_as_cat_list cols with imputed missing values
            X_NumericImputed (pd.Dataframe)             = dataframe having only num_list cols with imputed missing values
            main_df_NumericImputed (pd.Dataframe)       = dataframe having Numeric cols(now don't have values) + Categorical cols(having missing values)
        
        """
        
        # Return if not imputing
        if self.impute is False:
            return self
        
        if self.tunedRandomForest==True and self.verbose==True:
            print("\nNote: Missing value imputation is being performed using Gridsearch and Cross-validation of ML models.")
            print("      It may take longer time depending on size of data and number of column having missing values.\n\n")
            
        
        num_nans = X.isnull().sum().sum()
        num_total = X.shape[0] * X.shape[1]
        percentage_imputed = num_nans / num_total * 100
        
        # Grab list of object column names before doing imputation
        self.object_columns = X.select_dtypes(include=['object']).columns.values
        
        #Replacing None by NaN, if any None is present in X
        X.fillna( value=np.nan, inplace=True)
        
        # Seperating all columns into Categorical and Numeric
        cat_list=[]
        num_list=[]
        num_as_cat_list = self.numeric_columns_as_categorical
        
        #checking whether all the columns mentioned in num_as_cat_list are present in X or not
        if( num_as_cat_list is not None ):
            for col in num_as_cat_list:
                if( col not in list(X.columns) ):
                    raise HealthcareAIError('Column = {} mentioned in numeric_columns_as_categorical is not present in dataframe'.format(col))
    
        
        # segregating columns other than num_as_cat_list, into cat_list and num_list
        for c in X:
            if( (num_as_cat_list is None) or (num_as_cat_list is not None and c not in num_as_cat_list) ):
                if( X[c].dtype == np.dtype('O') or pd.api.types.is_categorical_dtype(X[c]) ):
                    cat_list.append( c )
                else:
                    num_list.append( c )
                    
        
        # Getting only num_as_cat_list columns with imputed missing values 
        # Also getNumericAsCategoricalImputedData() will internally fill the calculated imputation values along with column names in fill_dict
        if( num_as_cat_list is not None ):
            X_NumericAsCategoricalImputed = self.getNumericAsCategoricalImputedData( X = X[ num_as_cat_list ], 
                                                                       num_as_cat_list = num_as_cat_list        )
            X = X[ cat_list+num_list ].join( X_NumericAsCategoricalImputed, how='outer' ).copy()
            
        
        # Creating base copy of original Dataframe as 'main_df'
        main_df = X.copy()
        X_backup = X.copy()
        
        #--------------------------------------------------------------------------------------------------------------------------
        
        # Getting only Numeric columns with imputed missing values 
        # Also getNumericImputedData() will internally fill the predicted imputation values along with column names in fil_dict
        X_NumericImputed = self.getNumericImputedData( main_df=main_df.copy(), X=X.copy(), cat_list=cat_list, num_list=num_list )
        
        # main_df_NumericImputed = X_NumericImputed + CategoricalColumns
        main_df_NumericImputed = X_NumericImputed.join( main_df[ cat_list ], how='outer').copy()
        X_backup = main_df_NumericImputed.copy()
        X = main_df_NumericImputed.copy()
        
        # Getting only Categoric columns with imputed missing values
        # Also getCategoricalImputedData() will internally fill the predicted imputation values along with column names in fil_dict
        X_CategoricImputed = self.getCategoricalImputedData( main_df=main_df.copy(), X_NumericImputed=X_NumericImputed.copy(), X=X,                                                                      cat_list=cat_list, num_list=num_list )
        
        X = main_df.copy()
        
        #--------------------------------------------------------------------------------------------------------------------------
        
        if self.verbose:
            print("Percentage Imputed: %.2f%%" % percentage_imputed)
            print("Note: Impute will always happen on prediction dataframe, otherwise rows are dropped, and will lead "
                  "to missing predictions")
        

        # return self for scikit compatibility
        return self
    
    
    def transform(self, X, y=None):   
        """
        Description:
        ------------
            Missing value to be imputed are calculated using Mean and Mode of corresponding columns.
            1. Missing values of dataframe ae filled using self.fill_dict dictionary( updated in fill() function )
            2. Columns captured in 'self.object_columns' during fill() function are ensured to be of dtype='object'
            3. Columns specified in 'numeric_columns_as_categorical' are explicitly converted into dtype='object'
            
       """
                 
        # Return if not imputing
        if self.impute is False:
            return X
        
        #Replacing None by NaN, if any None is present in X
        X.fillna( value=np.nan, inplace=True)
        
            
        #Now filling up the missing values in X using fill_dict(which was updated in fit() function)
        for colName, imputeData in self.fill_dict.items():
            if( colName in X.columns ):
                X.loc[ X[ colName ].isnull(), colName ] = imputeData
        
        for i in self.object_columns:
                if X[i].dtype not in ['object', 'category']:
                    X[i] = X[i].astype('object')
                    
        if( self.numeric_columns_as_categorical is not None ):
            for col in self.numeric_columns_as_categorical:
                X[col] = X[col].astype( dtype='object', copy=True, error='raise' )
                
        return X
   
    
    
    def getNumericAsCategoricalImputedData( self, X, num_as_cat_list):
        """
        This function do below operations on num_as_cat_list columns:
            1. Calculate and impute the missing values using Mode value of each column.
            2. Update the calculated missing values in fill_dict. It will be used in transform() function.
        
        """
        for col in list( X.columns ):
            
            # If there is no null values in the column, skip current iteration
            if ( X[ col ].isnull().values.any()==False):
               continue
            
            # if column type is already categorical, raise exception
            if ( X[col].dtype == np.dtype('O') or pd.api.types.is_categorical_dtype(X[col]) ):
                raise HealthcareAIError("Column type of '{}' is already categorical, but it is mentioned in numeric_columns_as_categorical={}".format(col, self.numeric_columns_as_categorical) )
             
            
            imputeValue = X[col].value_counts().index[0]
            imputeData = np.array( object=[ imputeValue for i in range( X[col].isna().sum() ) ], dtype=np.int64 )
            
            self.fill_dict[col] = imputeData
            X[col].fillna( value=imputeValue, inplace=True )
            
            if self.verbose:
                print( "Column name                             =", col )
                print( "Total no of mising values               =", len(imputeData))
                percentage_missing = len(imputeData)/len(X[col])*100
                print( "Percentage missing values in the column = %.2f%% " % percentage_missing)
                print( "Top 10 predictions of missing values    =", imputeData[0:10] )
                print("------------------------------------------------------------------------------------------------")
        
        
        return X
                
    
    def getNumericImputedData( self, main_df, X, cat_list, num_list ):
        """
        Impute missing values in Numeric cols of dataframe.
        
        Args:
            main_df (pd.Dataframe) = copy of original dataframe(having missing values)
            X (pd.Dataframe)       = Dataframe on which operations will be performed
            cat_list (List of Strings) = List of categorical columns
            num_list (List of Strings) = List of numeric columns
       
        Local Vars:
            to_impute    (List)  = column in which missing values are to be imputed.
                                   Column name in list form is easy to use while indexing the dataframe.
                                   If at any place column name is required in String form we can use 'to_impute[0]' 
            to_impute[0] (String)= column name in which missing values are to be imputed
            all_columns (List)       = List of all columns
            predictor_columns (List) = List of columns to be used for predicting the missing values in to_impute[0] column
        
        Return:
            X_NumericImputed (pd.Dataframe) = have ONLY numeric colmns with imputed missing values
            Also fill_dict gets updated for each Numeric columns having missing values
            
        Steps:
            1. Whole process will run in a loop. This loop will run once for every col in num_list.
            2. That col will be the col under consideration in which mising values(if present) are to be imputed.
            3. find predictor_columns
            4. Temporarily impute missing values in predictor_columns using mean/mode as part of pre-data processing
               ( predictions will be done only for to_impute col.)
            5. Now append to_impute col to this dataframe
            6. Create dummy variables
            7. update the predictor_columns list
            8. Data pre-processing is completed, get the predictions of missing values in to_impute col
            9. Add the col name(to_impute) and predicted of missing values in fill_dict
            10. *** Also imputing the missing values of this column so that in next iteration there will 
                be lesser number of mising values in dataframe(X) that need to be temp imputed using
                mean/mode  (i.e step 4)
                
        """
        X_backup = X
        to_impute = []
        all_columns = []
        predictor_columns = []
        for i in num_list:
            X = X_backup
            to_impute = [i]

            # Check if to_impute col have any NaN Values. If no NaN values, no need to do imputation for this column
            if ( X[ to_impute ].isnull().values.any()==False):
                #print("No Nul values in = {} column, skipping the imputation loop".format( [i] ) )
                continue

            all_columns = list(X.columns)
            predictor_columns = all_columns.copy()
            predictor_columns.remove( to_impute[0] )

            X = X[ predictor_columns ]

            # Temporarily impute the missing values in X (using Mean and Median)
            # Note: After every iteration we will have 1 col less that is to be imputed using MeanMedian beacuse we are 
            #       imputing 1 columns per iteration using RandomForest  and adding it to X
            X = self.getTempImutedData( X )

            # As we didnt imputed mising values of to_impute col (since they are to be imputed using RandomForest)
            # Now joining to_impute col(having NaN values back to X)
            X = X.join( main_df[ to_impute ], how='outer')

            # Converting Categorical Cols --> to Numeric so that they can be feeded to ML model
            columns_to_dummify = list(X.select_dtypes(include=[object, 'category']))
            X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

            # Since some new cols are created after get_dummies, updating the predictor_columns List
            predictor_columns = list(X.columns)
            predictor_columns.remove(to_impute[0])
            
            # Get the predicted values for missing data
            y_pred_main = self.getImputePredictions(X=X, predictor_columns=predictor_columns, to_impute=to_impute, toImputeType='numeric' )
            
            # add the predicted imputation data to the fill_dict
            self.fill_dict[ to_impute[0] ] = y_pred_main

            # updating the imputed NaN of to_impute col in, X_backup
            # Now in next iteration this X_backup will be used as base DF
            X_backup.loc[ X_backup[ to_impute[0] ].isnull(), to_impute[0] ] = self.fill_dict[ to_impute[0] ]
            
            if self.verbose:
                print( "Column name                             =", to_impute[0] )
                print( "Total no of mising values               =", len(y_pred_main))
                percentage_missing = len(y_pred_main)/len(X_backup[to_impute[0]])*100
                print( "Percentage missing values in the column = %.2f%% " % percentage_missing)
                print( "Top 10 predictions of missing values    =", y_pred_main[0:10] )
                print("------------------------------------------------------------------------------------------------")
        
            
        X_NumericImputed = X_backup[ num_list ].copy()
        return X_NumericImputed


    def getCategoricalImputedData( self, main_df, X_NumericImputed, X, cat_list, num_list ):
        """
        Impute missing values in Numeric cols of dataframe.
        
        Args:
            main_df (pd.Dataframe) = copy of original dataframe(having missing values)
            X (pd.Dataframe)       = Dataframe on which operations will be performed
            cat_list (List of Strings) = List of categorical columns
            num_list (List of Strings) = List of numeric columns
       
        Local Vars:
            to_impute    (List)  = column in which missing values are to be imputed.
                                   Column name in list form is easy to use while indexing the dataframe.
                                   If at any place column name is required in String form we can use 'to_impute[0]' 
            to_impute[0] (String)= column name in which missing values are to be imputed
            all_columns (List)       = List of all columns
            predictor_columns (List) = List of columns to be used for predicting the missing values in to_impute[0] column
        
        Return:
            (pd.Dataframe) having ONLY categorical colmns with imputed missing values
            Also fill_dict is getting updated for each Numeric columns having missing values
        
         Steps:
            1. Whole process will run in a loop. This loop will run once for every col in num_list.
            2. That col will be the col under consideration in which mising values(if present) are to be imputed.
            3. find predictor_columns
            4. Temporarily impute missing values in predictor_columns using mean/mode as part of pre-data processing
               ( predictions will be done only for to_impute col.)
            5. Now join X_NumericImputed and to_impute col to this dataframe
            6. Create dummy variables( Also excluding the to_impute col, as it is also a categoric col )
            7. Update the predictor_columns list
            8. Since to_impute col is categorical, converting it into indexed form.
            9. Data pre-processing is completed, get the predictions of missing values in to_impute col
               Note: Here y_pred_main is in indexed form which needs to be converted back to original values.
            10. *** Imputing the missing values of this column so that in next iteration there will 
                be lesser number of mising values in dataframe(X) that need to be temp imputed using
                mean/mode  (i.e step 4)
                Also conveting back the indexed version of to_impute to its original categoric values.
            11. Add the col name(to_impute) and predicted missing values in fill_dicts
                
        """
        X_backup = X
        to_impute = []
        all_columns = []
        predictor_columns = []
        for i in cat_list:
            X = X_backup
            to_impute = [i]

            # Check if to_impute col have any NaN Values. If no NaN values, no need to do imputation for this column
            if ( X[ to_impute ].isnull().values.any()==False):
                #print("No Nul values in = {} column, skipping the imputation loop".format( [i] ) )
                continue

            all_columns = list(X.columns)
            predictor_columns = all_columns.copy()
            predictor_columns.remove( to_impute[0] )

            # tempImpute_columns = List of cols to be imputed temporarily using mode 
            # i.e ( cat_list - to_impute[0] )
            # to_impute is removed beacuse missing values of this column are to be perdicted using ML model
            # We are passing only cat_list cols to getTempImutedData() func and later joining it with X_NumericImputed df, so that                       getTempImutedData() func will take minimum time for temporary imputation and it wil not iterate on the columns whose missing               values are already imputed(i.e Numeric Cols)
            tempImpute_columns = cat_list.copy()
            tempImpute_columns.remove( to_impute[0] )
            
            X = X[ tempImpute_columns ]

            # Temporarily impute the missing values in X (using Mean and Mode)
            # Note: After every iteration we will have 1 col less that is to be imputed using MeanMode beacuse we are 
            #       imputing 1 columns per iteration using RandomForest and ading it to X
            X = self.getTempImutedData( X )
            
            # X = X(tempImpute_columns) + X_NumericImputed + main_df(to_impute)
            # As we didnt imputed mising values of to_impute col (since they are to be imputed using ML)
            # Now joining to_impute col(having NaN values back to X)
            X = X.join( X_NumericImputed, how='outer')
            X = X.join( main_df[ to_impute ], how='outer')

            # Converting Categorical Cols --> to indexed numeric so that they can be feeded to ML model
            # columns_to_dummify = columns_to_dummify - to_impute[0], as to_impute is a categorcal but it is to be imputed using ML model
            columns_to_dummify = list(X.select_dtypes(include=[object, 'category']))
            if( to_impute[0] in columns_to_dummify ):
                columns_to_dummify.remove( to_impute[0] )
            else:
                raise HealthcareAIError( "Col to_impute = {} not found in columns_to_dummify = {}".format( to_impute[0],                                                            str(columns_to_dummify) ) )
                
            X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

            # Since some new cols are created after get_dummies, updating the predictor_columns List
            predictor_columns = list(X.columns)
            predictor_columns.remove(to_impute[0])

            # Sice target col i.e to_impute[0] is a categorical feature, we have to convert it into indexed format(i.e 0,1,2)
            # from_List = original catregories ex.( A, B, C ... )
            # to_List   = indexed values       ex.( 0, 1, 2 ... )
            target_column = to_impute[0]
            from_List = list( X[target_column].unique() )   
            # removing NaN values from from_List( beacuse to_impute columns have missing values as well )
            if np.NaN in from_List:
                from_List.remove(np.NaN)
            elif (np.isnan( from_List ).any() ):
                #from_List.remove( np.NaN )
                from_List = [i for i in from_List if str(i) != 'nan']
            else:
                raise HealthcareAIError( "Null values didn't captured properly for col = {}, having unique values as = {}".format( to_impute[0], from_List ) )
            #Ensuring that each value in from_list is of type String (as to_impute is a categorical col and from_list have the unique values of to_impute col)
            from_List = list( map( str, from_List) )
            from_List.sort()
            #creating indexes version of from_List values
            to_List = [ i for i in range( 0,len(from_List) ) ]
            X[ target_column] = X[ target_column ].replace( from_List, to_List, inplace=False)

            # Get the predicted values for missing data
            y_pred_main = self.getImputePredictions( X=X, predictor_columns=predictor_columns, to_impute=to_impute, toImputeType='categorical' )
            
            
            # updating the imputed values of to_impute col in, X_backup
            # Now in next iteration this X_backup will be used as base Dataframe(X)
            X_backup.loc[ X_backup[ to_impute[0] ].isnull(), to_impute[0] ] = y_pred_main

            # Reconverting the idexed-to_impute column into its original form
            from_List, to_List = to_List, from_List
            X_backup[ to_impute] = X_backup[ to_impute ].replace( from_List, to_List, inplace=False)
            
            # add the imputation data to the fill_dict  
            # For that first we have to covert y_pred_main(indexed version. ex. 1,2..)  --> into actual version( ex. A, B...)
            y_pred_main_df = pd.DataFrame( data=y_pred_main, columns=to_impute )
            y_pred_main_df[ to_impute] = y_pred_main_df[ to_impute ].replace( from_List, to_List, inplace=False)            
            self.fill_dict[ to_impute[0] ] = y_pred_main_df[ to_impute[0] ].values
            
            if self.verbose:
                print( "Column name                             =", to_impute[0] )
                print( "Total no of mising values               =", len(y_pred_main))
                percentage_missing = len(y_pred_main)/len(X_backup[to_impute[0]])*100
                print( "Percentage missing values in the column = %.2f%% " % percentage_missing)
                print( "Top 10 predictions of missing values    =", y_pred_main_df[ to_impute[0] ].values[0:10] )
                print("------------------------------------------------------------------------------------------------")

            
        X_CategoricImputed = X_backup[ cat_list ].copy()
        return X_CategoricImputed


    def getTempImutedData( self, X ):
        """
        This function is used for temporary imputaion of mising values and impute missing values in a dataframe using Mean and Mode    . 
        
        Actual imputaion is done<in to_impute col> by doing prediction using ML model, but before creating ML model, during data pre-               pre-processing, there might be missing values in the other columns(i.e other than to_impute col), so for time being those missing           values are imputed using MeanMode Strategy
        
        Columns of dtype object or category (assumed categorical) = imputed with the mode (most frequent value in column).
        
        Columns of other types (assumed continuous)               = imputed with mean of column.
        """
        object_columns = X.select_dtypes(include=['object']).columns.values
        fill = pd.Series( [ X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') or pd.api.types.is_categorical_dtype(X[c])
                            else X[c].mean() 
                            for c in X
                          ]

                          , index=X.columns)
        result = X.fillna( fill )
        for i in object_columns:
                    if result[i].dtype not in ['object', 'category']:
                        result[i] = result[i].astype('object')
      
        return result.copy()
    
    
    def getImputePredictions( self, X, predictor_columns, to_impute, toImputeType ):
        """
        This method generate predictions of missing values
        
        Args:
            X (pd.Dataframe) = Inpute dataframe
            predictor_columns (List) = List of input columns for ML model
            to_impute (List)         = List<Although it will always have single element> of output column for ML model  
            toImputeType (String) = type of column to be imputed
        
        Return:
            y_pred_main (np.array) = predicted values for missing cells
        """
        
        # Seperating the mainDf into train(dont have NaN) and test(having NaN) data
        train = X[ X[to_impute[0]].isnull()==False ]
        test  = X[ X[to_impute[0]].isnull() ]

        # General X, y used for train test split
        # ***X_main = DF based on which we have to predict the NaN of to_impute col
        X = train[ predictor_columns ]
        y = train[ to_impute[0] ].values
        X_main = test[ predictor_columns ]

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100 )

        if( toImputeType=='numeric' ):
            algo = RandomForestRegressor( random_state=100 )
        elif( toImputeType=='categorical' ):
            algo = RandomForestClassifier( random_state=100 )
        else:
            raise HealthcareAIError("invalid toImputeType selected, select any of these : [ numeric, categorical ]")
                
        #tunedRandomForest = True
        if( self.tunedRandomForest==True ):
            algo = self.getTunedModel( baseModel=algo )
        
        fit_algo = algo.fit( X_train, y_train )
        #print( fit_algo.best_score_  )
        #print( fit_algo.best_params_ )
        #y_pred = fit_algo.predict( X_test )
        y_pred_main = fit_algo.predict( X_main )
        return y_pred_main.copy()

    def getTunedModel( self, baseModel ):
        n_estimators = [100, 200, 300, 400, 500]
        max_features = ['auto', 'sqrt']
        max_depth = [5, 10, 20, 30, 40, 50]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        #print(random_grid)
        
        model_tuned = RandomizedSearchCV(estimator = baseModel, param_distributions = random_grid, n_iter = 2, cv = 2, verbose=0, random_state=100 , n_jobs = -1)
        return model_tuned
        
  

        
        

######################################################################################################################################


class DataFrameConvertTargetToBinary(TransformerMixin):
    # TODO Note that this makes healthcareai only handle N/Y in pred column
    """
    Convert classification model's predicted col to 0/1 (otherwise won't work with GridSearchCV). Passes through data
    for regression models unchanged. This is to simplify the data pipeline logic. (Though that may be a more appropriate
    place for the logic...)

    Note that this makes healthcareai only handle N/Y in pred column
    """

    def __init__(self, model_type, target_column):
        self.model_type = model_type
        self.target_column = target_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO: put try/catch here when type = class and predictor is numeric
        # TODO this makes healthcareai only handle N/Y in pred column
        if self.model_type == 'classification':
            # Turn off warning around replace
            pd.options.mode.chained_assignment = None  # default='warn'
            # Replace 'Y'/'N' with 1/0
            X[self.target_column].replace(['Y', 'N'], [1, 0], inplace=True)

        return X


class DataFrameCreateDummyVariables(TransformerMixin):
    """Convert all categorical columns into dummy/indicator variables. Exclude given columns."""

    def __init__(self, excluded_columns=None):
        self.excluded_columns = excluded_columns

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        columns_to_dummify = list(X.select_dtypes(include=[object, 'category']))

        # remove excluded columns (if they are still in the list)
        for column in columns_to_dummify:
            if column in self.excluded_columns:
                columns_to_dummify.remove(column)

        # Create dummy variables
        X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

        return X


class DataFrameConvertColumnToNumeric(TransformerMixin):
    """Convert a column into numeric variables."""

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        X[self.column_name] = pd.to_numeric(arg=X[self.column_name], errors='raise')

        return X


class DataFrameUnderSampling(TransformerMixin):
    """
    Performs undersampling on a dataframe.
    
    Must be done BEFORE train/test split so that when we split the under/over sampled dataset.

    Must be done AFTER imputation, since under/over sampling will not work with missing values (imblearn requires target
     column to be converted to numerical values)
    """

    def __init__(self, predicted_column, random_seed=0):
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO how do we validate this happens before train/test split? Or do we need to? Can we implement it in the
        # TODO      simple trainer in the correct order and leave this to advanced users?

        # Extract predicted column
        y = np.squeeze(X[[self.predicted_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = X.drop([self.predicted_column], axis=1)

        # Initialize and fit the under sampler
        under_sampler = RandomUnderSampler(random_state=self.random_seed)
        x_under_sampled, y_under_sampled = under_sampler.fit_sample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_under_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_under_sampled = pd.Series(y_under_sampled)
        result[self.predicted_column] = y_under_sampled

        return result


class DataFrameOverSampling(TransformerMixin):
    """
    Performs oversampling on a dataframe.

    Must be done BEFORE train/test split so that when we split the under/over sampled dataset.

    Must be done AFTER imputation, since under/over sampling will not work with missing values (imblearn requires target
     column to be converted to numerical values)
    """

    def __init__(self, predicted_column, random_seed=0):
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO how do we validate this happens before train/test split? Or do we need to? Can we implement it in the
        # TODO      simple trainer in the correct order and leave this to advanced users?

        # Extract predicted column
        y = np.squeeze(X[[self.predicted_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = X.drop([self.predicted_column], axis=1)

        # Initialize and fit the under sampler
        over_sampler = RandomOverSampler(random_state=self.random_seed)
        x_over_sampled, y_over_sampled = over_sampler.fit_sample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_over_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_over_sampled = pd.Series(y_over_sampled)
        result[self.predicted_column] = y_over_sampled

        return result


class DataFrameDropNaN(TransformerMixin):
    """Remove NaN values. Columns that are NaN or None are removed."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Uses pandas.DataFrame.dropna function where axis=1 is column action, and
        # how='all' requires all the values to be NaN or None to be removed.
        return X.dropna(axis=1, how='all')


class DataFrameFeatureScaling(TransformerMixin):
    """Scales numeric features. Columns that are numerics are scaled, or otherwise specified."""

    def __init__(self, columns_to_scale=None, reuse=None):
        self.columns_to_scale = columns_to_scale
        self.reuse = reuse

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Check if it's reuse, if so, then use the reuse's DataFrameFeatureScaling
        if self.reuse:
            return self.reuse.fit_transform(X, y)

        # Check if we know what columns to scale, if not, then get all the numeric columns' names
        if not self.columns_to_scale:
            self.columns_to_scale = list(X.select_dtypes(include=[np.number]).columns)

        X[self.columns_to_scale] = StandardScaler().fit_transform(X[self.columns_to_scale])

        return X
