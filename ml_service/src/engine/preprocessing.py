import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

    def encode_experience_level(self, df, column='experience_level'):
        # ORDINAL ENCODING: Order matters
        mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
        df[column] = df[column].map(mapping)
        return df

    def encode_categories(self, df, column='preferred_topic'):
        # ONE-HOT ENCODING: Nominal data (no order)
        encoded_data = self.one_hot_encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded_data, columns=self.one_hot_encoder.get_feature_names_out([column]))
        return pd.concat([df.drop(column, axis=1), encoded_df], axis=1)

    def scale_numerical_features(self, df, columns=['study_hours']):
        # SCALING: Normalize data ranges
        df[columns] = self.scaler.fit_transform(df[columns])
        return df
