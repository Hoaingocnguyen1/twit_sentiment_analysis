Airflow khs lịch schedule bị lỗi hay j đó, sẽ p check lại _ Nho Duy Tran_ Test ca code retention data nx _ MLflow 
Dang làm code validate model để trước khi deploy, có vẻ bị lõio ở phần ạo run nên nó check vào 1 cái run mới ko tìm thấy model hoặc gì đó, ngủ dạy check lại r học tsa 
data artifact ném lên blob tải về chạy --sau 17 sẽ làm 

mlflow run ./mlflow  -e validate --no-conda -P model_name=distilbert-base-uncased  -P version=2 -P promote_to_prod=true -P min_accuracy=0.70     