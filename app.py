from flask import Flask,request,render_template
import os
from src.mlproject.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.mlproject.components.model_trainer import ModelTrainer,ModelTrainerConfig
from src.mlproject.components.data_transformation import DataTransformation

application=Flask(__name__)

app=application

## Route for a home page

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
            # Initialize DataTransformation
            data_transformation = DataTransformation()

            # Paths to the CSV files
            train_csv_path = os.path.join('artifact', 'train.csv')
            test_csv_path = os.path.join('artifact', 'test.csv')

            # Transform the data
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path=train_csv_path,
                test_path=test_csv_path
            )

            # Initialize ModelTrainer and train the model
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
    app.run(host="0.0.0.0",port=8080)