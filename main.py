import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class PatientInfo(BaseModel):
    gender: int
    age: int = None
    schooling: int
    breastfeeding: int = None
    varicella: int = None
    initial_symptom: int = None
    mono_or_polysymptomatic: int = None
    oligoclonal_bands: int = None
    llssep: int = None
    ulssep: int = None
    vep: int = None
    baep: int = None
    periventricular_mri: int
    cortical_mri: int
    infratentorial_mri: int
    spinal_cord_mri: int
    initial_edss: int = None
    final_edss: int = None




# with open('./ml_model/model_6.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
#     patient_info = {'gender': 1, 'age': 34, 'schooling': 20, 'breastfeeding': 1, 'varicella': 1,
#                     'initial_symptom': 2, 'mono_or_polysymptomatic': 1, 'oligoclonal_bands': 0, 'llssep': 1,
#                     'ulssep': 1, 'vep': 0, 'baep': 0, 'periventricular_mri': 0, 'cortical_mri': 1,
#                     'infratentorial_mri': 0, 'spinal_cord_mri': 1, 'initial_edss': 1, 'final_edss': 1, 'group': 1}
#     field_order = ['gender', 'varicella',
#                    'initial_symptom',
#                    'llssep', 'ulssep', 'vep', 'baep', 'periventricular_mri', 'cortical_mri',
#                    'infratentorial_mri', 'spinal_cord_mri', ]
#     data_list = np.array([patient_info[item] for item in field_order]).reshape(1, -1)
#     y_pred = loaded_model.predict(data_list)
#     print(y_pred)


# gender = {1:'Male', 2: 'Female'}
# breastfeeding = {1: 'yes', 2:'no', 3:'unknown'}
# varicella = {1 : 'positive', 2: 'negative', 3: 'unknown'}
# group = {1: 'CDMS' , 2: 'Non-CDMS' }

@app.get('/healthcheck')
async def root():
    return {'status': 'running'}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def say_hello(patient_info: PatientInfo):
    patient_info_dict = {k.lower(): v for k, v in patient_info.dict().items()}
    # use GaussianNB model
    with open('./ml_model/model_GaussianNB.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        # Format the data into two-dimensional data in the agreed order.
        # If necessary, you can discard some fields that are not helpful for prediction.
        field_order = ['gender', 'varicella',
                       'initial_symptom',
                       'llssep', 'ulssep', 'vep', 'baep', 'periventricular_mri', 'cortical_mri',
                       'infratentorial_mri', 'spinal_cord_mri', ]
        patient_info_list = np.array([patient_info_dict[item] for item in field_order]).reshape(1, -1)
        pred = loaded_model.predict(patient_info_list)[0]
    return {"is_positive": True if pred else False}


if __name__ == '__main__':
    print("hello")
