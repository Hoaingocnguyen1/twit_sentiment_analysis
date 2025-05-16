data artifact ném lên blob tải về chạy --sau 17 sẽ làm 
test rêtntion data, chạy nối được dag, code model thì ở notebook......
có thể cân nắc làm sạch code. CÒn vấn đề môi trường t cx kbiet, t đang để môi trg chạy airflow + mlflow vô dag luôn cho chung docker....... nhưng t k rõ sẽ tách hay để chung 2 môi trường này. có gì check giúp t..... Quan trọng nhất là code chạy đc dags + để cho cno làm dashboard. Mai thi ML xong sẽ tính nốt các phần còn lại 


mlflow run ./mlflow  -e validate --no-conda -P model_name=distilbert-base-uncased  -P version=2 -P promote_to_prod=true -P min_accuracy=0.70     