import optuna
import argparse

def study_to_csv(study, output_path):
    df = study.trials_dataframe()
    df.to_csv(output_path)


parser = argparse.ArgumentParser()
parser.add_argument('study-name', type=str, required=True)
parser.add_argument('storage', type=str, required=True)
parser.add_argument('output-path', type=str, required=True)

args = parser.parse_args()
study = optuna.load_study(study_name=args.study_name, storage=args.storage)
study_to_csv(study, args.output_path)
