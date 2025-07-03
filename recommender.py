import pandas as pd
import numpy as np
import joblib
from typing import List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

from schema import MobilePlanRequest, MobilePlan

class MobilePlanRecommender:
    def __init__(self, data_path: str):
        self.scaler = MinMaxScaler()
        self.data = pd.read_csv(data_path).replace({np.nan:None})
        self.cat_cols = ['with_phone', 'roam_asia', 'roam_world', 'caller_id_int']
        self.num_cols = ['price_monthly', 'contract_period', 'local_data', 'roam_data', 'talktime', 'sms']

    def fit(self):
        self.data['caller_id_int'] = self.data['caller_id'].astype(int)
        self.data['roam_asia'] = (~self.data['roam_data_region'].isna()).astype(int)
        self.data['roam_world'] = (self.data['roam_data_region'] == 'Worldwide').astype(int)
        self.data['with_phone'] = (self.data['plan_type'] == 'Phone').astype(int)

        df_num = pd.DataFrame(self.scaler.fit_transform(self.data[self.num_cols]), columns=self.num_cols)
        self.matrix = pd.concat([self.data[['plan_name'] + self.cat_cols], df_num], axis=1)
        
    def save(self, path_prefix: str):
        self.matrix.to_csv(f'{path_prefix}/matrix.csv', index=False)
        joblib.dump(self.scaler, f'{path_prefix}/scaler.pkl')
        
    def load(self, path_prefix: str):
        self.matrix = pd.read_csv(f'{path_prefix}/matrix.csv')
        self.scaler = joblib.load(f'{path_prefix}/scaler.pkl')

    def transform_request(self, request: MobilePlanRequest):
        req = pd.DataFrame([request.model_dump()])
        req['caller_id_int'] = req['caller_id'].astype(int)
        req['roam_asia'] = (~req['roam_data_region'].isna()).astype(int)
        req['roam_world'] = (req['roam_data_region'] == 'Worldwide').astype(int)
        req['with_phone'] = (req['plan_type'] == 'Phone').astype(int)
        
        return np.hstack([req[self.cat_cols].to_numpy(), self.scaler.transform(req[self.num_cols])])
    
    def recommend(self, request: MobilePlanRequest, k: int = 2) -> List[MobilePlan]:
        user_vector = self.transform_request(request)
        results = self.data.copy()
        results['distance'] = euclidean_distances(self.matrix[self.cat_cols + self.num_cols].to_numpy(), user_vector)
        output = results.copy()
        for key in request.model_dump(exclude_unset=True).keys():
            if key == 'price_monthly':
                output = output[output[key] <= getattr(request, key)]
            elif key in self.num_cols:
                output = output[output[key] >= getattr(request, key)]
            else:
                output = output[output[key] == getattr(request, key)]
        output = output.sort_values(by='distance', ascending=True).head(k)[['plan_name', 'plan_type', 'roam_data_region', 'caller_id'] + self.num_cols].to_dict(orient='records')
        return [MobilePlan(**plan) for plan in output]
    
    