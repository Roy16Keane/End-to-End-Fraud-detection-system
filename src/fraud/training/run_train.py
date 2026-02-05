from fraud.data.load_data import load_train_data
from fraud.training.train import train_final_model

def main():
    df = load_train_data("dataset/merged_data_pruned.parquet")
    meta = train_final_model(
        df_train=df,
        artifacts_dir="artifacts",
        model_dir="models",
    )
    print(" Training complete")
    print(meta)

if __name__ == "__main__":
    main()
