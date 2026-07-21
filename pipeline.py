from src.clean_data import clean_data
from src.import_data import importing_data
from src.feature_engineering import feature_engineer


def main():
    importing_data()
    clean_data()
    feature_engineer()


if __name__ == "__main__":
    main()
