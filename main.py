from PipelineConfiguration import PipelineConfiguation
from Pipeline import Pipeline

three_model_weights  = [[0.05, 0.05, 0.9],
                        [0.1, 0.1, 0.8],
                        [0.15, 0.15, 0.7],
                        [0.2, 0.2, 0.6],
                        [0.25, 0.25, 0.5]]

two_model_weights = [[0.1, 0.9],
                     [0.2, 0.8],
                     [0.3, 0.7],
                     [0.4, 0.6],
                     [0.5, 0.5]]

#Ensemble lists
two_model_ensemble_list = [['rf', 'inception'],
                           ['svc', 'inception'],
                           ['inception', 'effinet'],
                           ['rf', 'effinet'],
                           ['svc', 'effinet']]

three_model_ensemble_list = [['cnn', 'effinet', 'inception'],
                             ['rf', 'svc', 'inception'],
                             ['rf', 'svc', 'effinet']]

class_list = ['Beauty', 'Family', 'Fashion', 'Fitness', 'Food', 'Interior','Pet', 'Travel']

data_dir = r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\Dataset'
output_dir = r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\Results\\'

def main():
    config = PipelineConfiguation(class_list, data_dir, output_dir, two_model_ensemble_list, two_model_weights, three_model_ensemble_list, three_model_weights)
    pipeline = Pipeline(config)
    results, two_model_results, three_model_results = pipeline.run()
    return results, two_model_results, three_model_results

results = main()
