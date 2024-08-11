from flask import Flask,request,render_template
import os
from src.mlproject.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.mlproject.components.model_trainer import ModelTrainer,ModelTrainerConfig
from src.mlproject.components.data_transformation import DataTransformation

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        try:
            data_transformation = DataTransformation()
 
            train_csv_path = os.path.join('artifact', 'train.csv')
            test_csv_path = os.path.join('artifact', 'test.csv')

            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path=train_csv_path,
                test_path=test_csv_path
            )

            model_trainer = ModelTrainer()
            model_trainer_config=ModelTrainerConfig()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            best_model = model_trainer_config.best_model_name
            
            status = "Training Completed Successfully"

            return render_template('train.html', r2_score=r2_score, best_model=best_model, status=status)
        except Exception as e:
            return render_template('train.html', r2_score="N/A", best_model="N/A", status=str(e))
    else:
        return render_template('train.html', r2_score="N/A", best_model="N/A", status="")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)        # 127.0.0.1:5000 if not visible link use that #debug=True
    
    
    
    # from src.mlproject.logger import logging
# from src.mlproject.exception import CustomException
# from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig 
# from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
# from src.mlproject.components.model_trainer import ModelTrainer,ModelTrainerConfig
# import sys

# # if __name__=="__ main__ ":
# logging.info("The execution has started")
# try:
#     #data_ingestion_config=DataIngestionConfig()
#     data_ingestion=DataIngestion ()
#     train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
    
#     # data_transformation_config=DataIngestionConfig()
#     data_transformation=DataTransformation()
#     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    
#     model_trainer=ModelTrainer()
#     model_trainer.initiate_model_trainer(train_arr,test_arr)
# except Exception as e:
#     logging.info("Custom Exception")
#     raise CustomException(e,sys)