
# Predicting Students Performance in Exams using machine learning


This project focuses on predicting the student marks based on various other factor/properties.The dataset used in this project is the "Students Performance in Exams" dataset, which is publicly available. The classes are ordered and not balanced .Your task is to predict the maths marks using the given data.


# Dataset Link

https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data


## Run Locally

Clone the project

```bash
    git clone https://github.com/kunalshelke90/Students_Marks_Prediction_in_Exams.git
```

Go to the project directory

```bash
    cd Students_Marks_Prediction_in_Exams
```

Create a virtual environment and install dependencies:

```bash
    conda create -p myenv python=3.8 -y
```


```bash
   conda activate myenv
```
```bash
    pip install -r requirements.txt
```

## Usage
1. Start the Flask application:

```bash
    python app.py
```
2. Access the application:
Open your web browser and go to http://localhost:8080 to interact with the application. or http://127.0.0.1:8080

## Setup github secrets:

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo link-  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = 

# Features :
- Predicts student performance in mathematics based on multiple factors.
- Provides insights into the influence of gender, ethnicity, parental level of education, lunch type, and test preparation course on student performance.
- User-friendly interface for inputting student information and obtaining predictions.

# Content and Key identifiers

Gender: Female, Male

Race/Ethnicity: Grouped A - E

Parental Level of Education: Bachelor's Degree, Some College, Master's Degree, Associate's Degree, High School, Some High School

Lunch: Standard, Free/Reduced

Test preparation course: Completed, None

Math Score

Reading Score

Writing Score

# Customization 

Data Ingestion: Customize data_ingestion.py in the src/mlproject/components folder to suit your data source and schema. You can modify the connection settings for your Cassandra database and adjust the data loading logic in src/mlproject/utils.py .

Data Transformation: Modify data_transformation.py in the src/mlproject/components folder to apply different scaling methods, feature engineering techniques, or transformations according to your dataset's needs.

Model Training: Customize model_training.py in the src/mlproject/components folder to experiment with different models, hyperparameters, and evaluation metrics. You can also integrate other ML libraries like TensorFlow or PyTorch.

Web Interface: Modify the HTML templates in the templates/ folder to match your preferred UI design. You can add or remove input fields, change styles, and customize the prediction output format.

# License

This project is licensed under the MIT License. See the LICENSE file for details
