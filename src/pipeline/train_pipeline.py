from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline() -> tuple[float, float, float]:
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    model_trainer = ModelTrainer()
    return model_trainer.initiate_model_training(train_arr, test_arr)


if __name__ == "__main__":
    mae, rmse, r2 = run_training_pipeline()
    print(f"Training complete | MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
