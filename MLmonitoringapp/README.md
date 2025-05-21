ML Monitoring Azure Function App
Ứng dụng Azure Function tự động theo dõi hiệu suất của mô hình ML và huấn luyện lại khi cần thiết.
Tổng quan
Ứng dụng này thực hiện các chức năng sau:

Quét Blob Storage mỗi 3 ngày để tìm các file dữ liệu mới (định dạng clean_YYYYMMDD_iii)
Xử lý các file dữ liệu trong vòng 5 ngày gần nhất
Đánh giá hiệu suất của mô hình ML hiện tại với dữ liệu mới
Tự động huấn luyện lại mô hình nếu hiệu suất giảm dưới ngưỡng
Đăng ký mô hình mới vào Azure ML Registry nếu hiệu suất được cải thiện

Kiến trúc
Ứng dụng bao gồm 4 Azure Function:

Blob Scanner Trigger: Function chạy theo lịch mỗi 3 ngày, quét Blob Storage để tìm các file dữ liệu mới và gửi chúng vào queue xử lý.
File Processor: Function được kích hoạt khi có message trong queue, tải nội dung file từ Blob Storage, xử lý và chuẩn bị cho việc đánh giá.
Model Evaluator: Function đánh giá hiệu suất của mô hình hiện tại với dữ liệu mới, quyết định có cần huấn luyện lại không.
Model Trainer: Function huấn luyện mô hình mới và đăng ký vào Azure ML Registry nếu hiệu suất được cải thiện.

Cấu hình
Ứng dụng sử dụng các cấu hình sau trong file local.settings.json:

STORAGE_CONNECTION_STRING: Connection string để kết nối với Azure Storage
DATA_CONTAINER_NAME: Tên container chứa dữ liệu
DATA_PROCESSING_QUEUE: Tên queue để xử lý dữ liệu
EVALUATION_THRESHOLD: Ngưỡng để đánh giá hiệu suất mô hình (mặc định: 0.75)
DAYS_TO_SCAN: Số ngày dữ liệu cần xử lý (mặc định: 5)
MODEL_REGISTRY_NAME: Tên registry để lưu mô hình
AZURE_SUBSCRIPTION_ID: ID của Azure Subscription
AZURE_RESOURCE_GROUP: Tên Resource Group
AZURE_ML_WORKSPACE: Tên Azure ML Workspace

Triển khai
Yêu cầu

Azure Subscription
Azure Storage Account
Azure ML Workspace
Python 3.8 hoặc cao hơn

Các bước triển khai

Cài đặt Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

Cài đặt các gói phụ thuộc
pip install -r requirements.txt

Cấu hình local.settings.json với các thông tin kết nối
Chạy ứng dụng cục bộ
func start

Triển khai lên Azure
func azure functionapp publish <app-name>


Luồng dữ liệu

Blob Scanner Trigger (Chạy mỗi 3 ngày)

Quét Blob Storage để tìm các file với định dạng clean_YYYYMMDD_iii
Gửi thông tin file vào queue data-processing-queue


File Processor (Xử lý các message trong queue)

Tải nội dung file từ Blob Storage
Xử lý và chuẩn bị dữ liệu
Gửi thông tin vào queue model-evaluation-queue


Model Evaluator (Đánh giá hiệu suất)

Lấy mô hình mới nhất từ registry
Đánh giá hiệu suất với dữ liệu mới
Nếu hiệu suất dưới ngưỡng, gửi vào queue model-training-queue


Model Trainer (Huấn luyện lại mô hình)

Huấn luyện mô hình mới
Đánh giá hiệu suất
Đăng ký vào registry nếu hiệu suất được cải thiện



Lưu ý và Tùy chỉnh

Định dạng file dữ liệu: Mặc định, ứng dụng sẽ xử lý các file có định dạng clean_YYYYMMDD_iii (ví dụ: clean_20250521_001).
Thuật toán mô hình: Mặc định sử dụng RandomForestClassifier. Có thể tùy chỉnh trong model_helper.py.
Đánh giá hiệu suất: Mặc định sử dụng F1 score để đánh giá. Có thể tùy chỉnh trong model_helper.py.
Ngưỡng đánh giá: Mặc định là 0.75. Có thể tùy chỉnh trong cấu hình.