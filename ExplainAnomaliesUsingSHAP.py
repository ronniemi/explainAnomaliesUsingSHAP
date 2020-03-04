import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import shap
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger('shap')
logger.disabled = True


class ExplainAnomaliesUsingSHAP:
    '''
    This class implements method described in 'Explaining Anomalies Detected by Autoencoders Using SHAP' to explain
    anomalies revealed by an unsupervised Autoencoder model using SHAP.
    '''

    autoencoder = None
    num_anomalies_to_explain = None
    reconstruction_error_percent = None
    shap_values_selection = None
    counter = None

    def __init__(self, num_anomalies_to_explain=100, reconstruction_error_percent=0.5, shap_values_selection='mean'):
        """
        Args:
            num_anomalies_to_explain (int): number of top ranked anomalies (ranked by anomaly score that is the mse) to
                                            explain.
            reconstruction_error_percent (float): Number between 0 to 1- see explanation to this parameter in
                                                  'Explaining Anomalies Detected by Autoencoders Using SHAP' under
                                                  ReconstructionErrorPercent.
            shap_values_selection (str): One of the possible methods to choose explaining features by their SHAP values.
                                         Can be: 'mean', 'median', 'constant'. See explanation to this parameter in
                                         'Explaining Anomalies Detected by Autoencoders Using SHAP' under
                                         SHAPvaluesSelection.
        """

        self.num_anomalies_to_explain = num_anomalies_to_explain
        self.reconstruction_error_percent = reconstruction_error_percent
        self.shap_values_selection = shap_values_selection

    def train_model(self, x_train, nb_epoch=1000, batch_size=64):
        """
        Train 6-layer Autoencoder model on the given x_train data.

        Args:
            x_train (data frame): The data to train the Autoencoder model on
            nb_epoch (int): Number of epoch the model will perform
            batch_size (int): Size of each batch of data enter to the model

        Returns:
            model: Trained autoencoder
        """
        
        input_dim = x_train.shape[1]

        input_layer = Input(shape=(input_dim,))

        encoder = Dense(int(input_dim / 2), activation="relu", activity_regularizer=regularizers.l1(10e-7))(
            input_layer)

        encoder = Dense(int(input_dim / 4), activation="relu", kernel_regularizer=regularizers.l2(10e-7))(encoder)

        decoder = Dense(int(input_dim / 2), activation='relu', kernel_regularizer=regularizers.l2(10e-7))(encoder)

        decoder = Dense(input_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(10e-7))(decoder)

        self.autoencoder = Model(inputs=input_layer, outputs=decoder)

        self.autoencoder.summary()

        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        self.autoencoder.fit(x_train, x_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True,
                             validation_split=0.1, verbose=2, callbacks=[earlystopper])

        return self.autoencoder

    def get_top_anomaly_to_explain(self, x_explain):
        """
        Sort all records in x_explain by their MSE calculated according to their prediction by the trained Autoencoder
        and return the top num_anomalies_to_explain (its value given by the user at class initialization) records.

        Args:
            x_explain (data frame): Set of records we want to explain the most anomalous ones from it.

        Returns:
            list: List of index of the top num_anomalies_to_explain records with highest MSE that will be explained.
        """
        
        predictions = self.autoencoder.predict(x_explain)
        square_errors = np.power(x_explain - predictions, 2)
        mse_series = pd.Series(np.mean(square_errors, axis=1))

        most_anomal_trx = mse_series.sort_values(ascending=False)
        columns = ["id", "mse_all_columns"]
        columns.extend(["squared_error_" + x for x in list(x_explain.columns)])
        items = []
        for x in most_anomal_trx.iteritems():
            item = [x[0], x[1]]
            item.extend(square_errors.loc[x[0]])
            items.append(item)

        df_anomalies = pd.DataFrame(items, columns=columns)
        df_anomalies.set_index('id', inplace=True)

        top_anomalies_to_explain = df_anomalies.head(self.num_anomalies_to_explain).index
        return top_anomalies_to_explain

    def get_num_features_with_highest_reconstruction_error(self, total_squared_error, errors_df):
        """
        Calculate the number of features whose reconstruction errors sum to reconstruction_error_percent of the
        total_squared_error of the records that selected to be explained at the moment. This is the number of the
        top reconstructed errors features that going to be explained and eventually this features together with their
        explanation will build up the features explanation set of this record.

        Args:
            total_squared_error (int): MSE of the records selected to be explained
            errors_df (data frame): The reconstruction error of each feature- this is the first output output of
                                    get_errors_df_per_record function

        Returns:
            int: Number of features whose reconstruction errors sum to reconstruction_error_percent of the
                 total_squared_error of the records that selected to be explained at the moment
        """

        error = 0
        for num_of_features, index in enumerate(errors_df.index):
            error += errors_df.loc[index, 'err']
            if error >= self.reconstruction_error_percent * total_squared_error:
                break
        return num_of_features + 1

    def get_background_set(self, x_train, background_size=200):
        """
        Get the first background_size records from x_train data and return it. Used for SHAP explanation process.

        Args:
            x_train (data frame): the data we will get the background set from
            background_size (int): The number of records to select from x_train. Default value is 200.

        Returns:
            data frame: Records from x_train that will be the background set of the explanation of the record that we
                        explain at that moment using SHAP.
        """

        background_set = x_train.head(background_size)
        return background_set

    def get_errors_df_per_record(self, record):
        """
        Create data frame of the reconstruction errors of each features of the given record. Eventually we get data
        frame so each row contain the index of feature, its name, and its reconstruction error based on the record
        prediction provided by the trained autoencoder. This data frame is sorted by the reconstruction error of the
        features

        Args:
            record (pandas series): The record we explain at the moment; values of all its features.

        Returns:
            data frame: Data frame of all features reconstruction error sorted by the reconstruction error.
        """

        prediction = self.autoencoder.predict(np.array([[record]])[0])[0]
        square_errors = np.power(record - prediction, 2)
        errors_df = pd.DataFrame({'col_name': square_errors.index, 'err': square_errors}).reset_index(drop=True)
        total_mse = np.mean(square_errors)
        errors_df.sort_values(by='err', ascending=False, inplace=True)
        return errors_df, total_mse

    def get_highest_shap_values(self, shap_values_df):
        """
        Choosing explaining features based on their SHAP values by shap_values_selection method (mean, median, constant)
        i.e. remove all features with SHAP values that do not meet the method requirements as described in 'Explaining
        Anomalies Detected by Autoencoders Using SHAP' under SHAPvaluesSelection.

        Args:
            shap_values_df (data frame): Data frame with all existing features and their SHAP values.

        Returns:
            data frame: Data frame that contain for each feature we explain (features with high reconstruction error)
                        its explaining features that selected by the shap_values_selection method and their SHAP values.
        """

        all_explaining_features_df = pd.DataFrame()

        for i in range(shap_values_df.shape[0]):
            shap_values = shap_values_df.iloc[i]

            if self.shap_values_selection == 'mean':
                treshold_val = np.mean(shap_values)

            elif self.shap_values_selection == 'median':
                treshold_val = np.median(shap_values)

            elif self.shap_values_selection == 'constant':
                num_explaining_features = 5
                explaining_features = shap_values_df[i:i + 1].stack().nlargest(num_explaining_features)
                all_explaining_features_df = pd.concat([all_explaining_features_df, explaining_features], axis=0)
                continue

            else:
                raise ValueError('unknown SHAP value selection method')

            num_explaining_features = 0
            for j in range(len(shap_values)):
                if shap_values[j] > treshold_val:
                    num_explaining_features += 1
            explaining_features = shap_values_df[i:i + 1].stack().nlargest(num_explaining_features)
            all_explaining_features_df = pd.concat([all_explaining_features_df, explaining_features], axis=0)
        return all_explaining_features_df

    def func_predict_feature(self, record):
        """
        Predict the value of specific feature (with 'counter' index) using the trained autoencoder

        Args:
            record (pandas series): The record we explain at the moment; values of all its features.

        Returns:
            list: List the size of the number of features, contain the value of the predicted features with 'counter'
                  index (the feature we explain at the moment)
        """

        record_prediction = self.autoencoder.predict(record)[:, self.counter]
        return record_prediction

    def explain_unsupervised_data(self, x_train, x_explain, autoencoder=None, return_shap_values=False):
        """
        First, if Autoencoder model not provided ('autoencoder' is None) train Autoencoder model on given x_train data.
        Then, for each record in 'top_records_to_explain' selected from given 'x_explain' as described in
        'get_top_anomaly_to_explain' function, we use SHAP to explain the features with the highest reconstruction
        error based on the output of 'get_num_features_with_highest_reconstruction_error' function described above.
        Then, after we got the SHAP value of each feature in the explanation of the high reconstructed error feature,
        we select the explaining features using 'highest_contributing_features' function described above. Eventually,
        when we got the explaining features for each one of the features with highest reconstruction error, we build the
        explaining features set so the feature with the highest reconstruction error and its explaining features enter
        first to the explaining features set, then the next feature with highest reconstruction error and its explaining
        features enter to the explaining features set only if they don't already exist in the explaining features set
        and so on (full explanation + example exist in 'Explaining Anomalies Detected by Autoencoders Using SHAP')

        Args:
            x_train (data frame): The data to train the autoencoder model on and to select the background set from (for
                                  SHAP explanation process)
            x_explain (data frame): The data from which the top 'num_anomalies_to_explain' records are selected by their
                                    MSE to be explained.
            autoencoder (model): Trained Autoencoder model that will be used to explain x_explain data. If None (model 
                                 not provided) then we will build and train from scratch a Autoencoder model as described 
                                 in train_model function.
            return_shap_values (bool): If False, the resulting explnation featues set for each record will include only 
                                       the names of the explaining features. If True, in addition to explaining feature name,
                                       the explnation featues set will include the SHAP value of each feature in the explnation
                                       featues set so the explnation featues set will be composed of tupels of (str, float)
                                       when str will be the name of the explaining feature and float will be its SHAP value.
                                       Note that for the explained features (features with high reconstraction error), if they 
                                       did not appear in previuse feature explanation (explnation of feature with higher 
                                       recustraction error), they will not have any SHAP values. Therefore they get unique
                                       value of -1.
                                       
        Returns:
            dict: Return all_sets_explaining_features dictionary that contain the explanation for
                  'top_records_to_explain' records so that the keys are int; the records indexes and the values are
                  lists; the explanation features sets.
        """
        
        self.autoencoder = autoencoder
        if self.autoencoder is None:
            self.train_model(x_train)      
        
        top_records_to_explain = self.get_top_anomaly_to_explain(x_explain)
        all_sets_explaining_features = {}

        for record_idx in top_records_to_explain:
            print(record_idx)

            record_to_explain = x_explain.loc[record_idx]

            df_err, total_mse = self.get_errors_df_per_record(record_to_explain)
            num_of_features = self.get_num_features_with_highest_reconstruction_error(total_mse * df_err.shape[0],
                                                                                      df_err)

            df_top_err = df_err.head(num_of_features)
            all_sets_explaining_features[record_idx] = []
            shap_values_all_features = [[] for num in range(num_of_features)]

            backgroungd_set = self.get_background_set(x_train, 200).values
            for i in range(num_of_features):
                self.counter = df_top_err.index[i]
                explainer = shap.KernelExplainer(self.func_predict_feature, backgroungd_set)
                shap_values = explainer.shap_values(record_to_explain, nsamples='auto')
                shap_values_all_features[i] = shap_values

            shap_values_all_features = np.fabs(shap_values_all_features)

            shap_values_all_features = pd.DataFrame(data=shap_values_all_features, columns=x_train.columns)
            highest_contributing_features = self.get_highest_shap_values(shap_values_all_features)
            
            for idx_explained_feature in range(num_of_features):
                set_explaining_features =[]
                for idx, row in highest_contributing_features.iterrows():
                    if idx[0] == idx_explained_feature:
                        set_explaining_features.append((idx[1], row[0]))
                explained_feature_index = df_top_err.index[idx_explained_feature]
                set_explaining_features.insert(0, (x_train.columns[explained_feature_index], -1))

                all_sets_explaining_features[record_idx].append(set_explaining_features)

            final_set_features = []
            final_set_items = []
            for item in sum(all_sets_explaining_features[record_idx], []):
                if item[0] not in final_set_features:
                    final_set_features.append(item[0])
                    final_set_items.append(item)
                    
            if return_shap_values:
                all_sets_explaining_features[record_idx] = final_set_items
            else:
                all_sets_explaining_features[record_idx] = final_set_features

        return all_sets_explaining_features
