import mlflow
import os
import logging
import torch
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.models.transformer_manager import ModelManager

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Class quản lý tương tác với MLflow Model Registry
    """
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Khởi tạo kết nối với MLflow tracking server
        
        Args:
            tracking_uri: URL của MLflow tracking server (tùy chọn)
        """
        # Thiết lập tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        else:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Khởi tạo client
        try:
            self.client = mlflow.tracking.MlflowClient()
            logger.info("MLflow client được khởi tạo thành công.")
        except Exception as e:
            logger.error(f"Không thể khởi tạo MLflow client: {e}")
            raise

    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        model_manager: Optional[ModelManager] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Đăng ký model vào MLflow Model Registry
        
        Args:
            run_id: MLflow run ID
            model_name: Tên model để đăng ký
            description: Mô tả model (tùy chọn)
            metrics: Dictionary chứa các metrics (tùy chọn)
            model_manager: Instance của ModelManager để lưu artifacts (tùy chọn)
            tags: Dictionary chứa các tags (tùy chọn)
            
        Returns:
            Version của model đã đăng ký
        """
        try:
            uri = f"runs:/{run_id}/model"
            details = mlflow.register_model(model_uri=uri, name=model_name)
            version = details.version

            # Cập nhật mô tả
            if description:
                self.client.update_registered_model(name=model_name, description=description)
                logger.info(f"Đã cập nhật mô tả cho model '{model_name}'")

            # Thêm tags
            if tags:
                for tag_key, tag_value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=tag_key,
                        value=tag_value
                    )
                logger.info(f"Đã thêm {len(tags)} tags vào model '{model_name}' phiên bản {version}")

            # Log metrics
            if metrics:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.client.log_metric(run_id, metric_name, value)
                logger.info(f"Đã log {len(metrics)} metrics vào run {run_id}")

            # Lưu model artifacts
            if model_manager:
                self.save_model_artifacts(model_manager, run_id)

            logger.info(f"Model '{model_name}' đã được đăng ký với phiên bản {version}")
            return version

        except Exception as e:
            logger.error(f"Lỗi khi đăng ký model: {e}")
            raise

    def save_model_artifacts(self, model_manager: ModelManager, run_id: str, task: str = "text-classification") -> None:
        """
        Lưu model artifacts vào MLflow run
        
        Args:
            model_manager: Instance của ModelManager chứa model và tokenizer
            run_id: MLflow run ID
            task: Loại task của model (mặc định: "text-classification")
        """
        try:
            # Lưu model và tokenizer
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model_manager.model,
                    "tokenizer": model_manager.tokenizer
                },
                artifact_path="model",
                run_id=run_id,
                task=task
            )

            # Lưu config model nếu có
            if hasattr(model_manager.model, "config"):
                config_path = os.path.join(model_manager.output_dir, "config_model.json")
                model_manager.model.config.to_json_file(config_path)
                mlflow.log_artifact(config_path, run_id=run_id)

            logger.info(f"Đã lưu model artifacts cho run {run_id}")

        except Exception as e:
            logger.error(f"Lỗi khi lưu model artifacts: {e}")
            raise

    def add_aliases(self, model_name: str, version: str, aliases: List[str]) -> None:
        """
        Thêm aliases cho phiên bản model đã đăng ký
        
        Args:
            model_name: Tên của model đã đăng ký
            version: Phiên bản của model
            aliases: Danh sách aliases cần thêm
        """
        try:
            for alias in aliases:
                try:
                    self.client.set_registered_model_alias(model_name, alias, version)
                    logger.info(f"Đã thêm alias '{alias}' vào model {model_name} v{version}")
                except Exception as alias_error:
                    logger.warning(f"Không thể thêm alias '{alias}': {alias_error}")
            
            logger.info(f"Đã thêm {len(aliases)} aliases vào model {model_name} v{version}")
        except Exception as e:
            logger.error(f"Lỗi khi thêm aliases: {e}")
            raise

    def load_model(
            self,
            model_name: str,
            model_path: Optional[str] = None,
            version: Optional[str] = None,
            stage: str = "Production"
        ) -> "ModelManager":
        # 1) Nếu có đường dẫn cục bộ, tải trực tiếp
        if model_path:
            logger.info(f"Đang tải model từ đường dẫn cục bộ: {model_path}")
            model_obj = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
                device_map='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
            tokenizer_obj = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            return ModelManager(
                model_name=model_name,
                model_path=model_path,
                finetune=False,
                model_obj=model_obj,
                tokenizer_obj=tokenizer_obj
            )

        # 2) Nếu không, tải từ MLflow model registry
        model_uri = f"models:/{model_name}/{version or stage}"
        logger.info(f"Đang tải từ MLflow URI: {model_uri}")

        try:
            loaded = mlflow.transformers.load_model(model_uri)
        except ValueError as e:
            if "Repo id must be in the form" in str(e):
                logger.warning("Đang thử lại với repo_type='model'")
                loaded = mlflow.transformers.load_model(model_uri, repo_type="model")
            else:
                logger.error(f"Tải model thất bại: {e}")
                raise

        # Trích xuất model và tokenizer
        if isinstance(loaded, dict):
            model_obj = loaded.get("model")
            tokenizer_obj = loaded.get("tokenizer")
        elif hasattr(loaded, "model") and hasattr(loaded, "tokenizer"):
            model_obj = loaded.model
            tokenizer_obj = loaded.tokenizer
        else:
            raise RuntimeError("Đối tượng đã tải không chứa model và tokenizer")

        # Đóng gói trong ModelManager và trả về
        return ModelManager(
            model_path=model_uri,
            finetune=False,
            model_obj=model_obj,
            tokenizer_obj=tokenizer_obj
        )

    def list_models(self) -> List[Dict]:
        """
        Liệt kê tất cả các models trong registry
        
        Returns:
            Danh sách các models với thông tin chi tiết
        """
        try:
            regs = self.client.search_registered_models()
            return [
                {
                    'name': m.name,
                    'latest_versions': [
                        {
                            'version': v.version,
                            'status': v.status,
                            'run_id': v.run_id,
                            'metrics': self.client.get_run(v.run_id).data.metrics
                        }
                        for v in m.latest_versions
                    ]
                }
                for m in regs
            ]
        except Exception as e:
            logger.error(f"Lỗi khi liệt kê models: {e}")
            raise

    def get_latest_version(self, model_name: str) -> str:
        """
        Lấy phiên bản mới nhất của model
        
        Args:
            model_name: Tên của model
            
        Returns:
            Phiên bản mới nhất của model
        """
        try:
            reg = self.client.get_registered_model(model_name)
            return reg.latest_versions[0].version
        except Exception as e:
            logger.error(f"Lỗi khi lấy phiên bản mới nhất: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """
        Chuyển đổi stage của model
        
        Args:
            model_name: Tên của model
            version: Phiên bản model
            stage: Stage mới (ví dụ: "Production", "Staging", "Archived")
        """
        try:
            self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
            logger.info(f"Đã chuyển {model_name} v{version} sang stage {stage}")
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi stage của model: {e}")
            raise

    def delete_model_version(self, model_name: str, version: str) -> None:
        """
        Xóa phiên bản model
        
        Args:
            model_name: Tên của model
            version: Phiên bản model cần xóa
        """
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Đã xóa {model_name} v{version}")
        except Exception as e:
            logger.error(f"Lỗi khi xóa phiên bản model: {e}")
            raise
            
    def get_model_version_by_alias(self, model_name: str, alias: str) -> Dict[str, Any]:
        """
        Lấy thông tin về phiên bản model thông qua alias
        
        Args:
            model_name: Tên của model
            alias: Alias của phiên bản model (ví dụ: "latest", "stable", "champion", v.v.)
            
        Returns:
            Dictionary chứa thông tin về phiên bản model
        """
        try:
            # Lấy version dựa trên alias
            version = self.client.get_model_version_by_alias(name=model_name, alias=alias)
            
            # Lấy thêm thông tin run_id để truy cập metrics
            run_id = version.run_id
            run = self.client.get_run(run_id) if run_id else None
            
            # Xây dựng và trả về thông tin chi tiết
            result = {
                'name': model_name,
                'version': version.version,
                'alias': alias,
                'status': version.status,
                'run_id': run_id,
                'creation_timestamp': version.creation_timestamp
            }
            
            # Thêm metrics nếu có
            if run and hasattr(run, 'data') and hasattr(run.data, 'metrics'):
                result['metrics'] = run.data.metrics
                
            logger.info(f"Đã truy vấn thông tin cho {model_name} với alias '{alias}' (version: {version.version})")
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin model qua alias: {e}")
            raise