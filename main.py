from lottery_data import LotteryData
from lottery_predictor_svr import LotteryPredictorSVR
from lottery_predictor_rf import LotteryPredictorRF
from lottery_predictor_knn import LotteryPredictorKNN
from lottery_predictor_gbr import LotteryPredictorGBR
from lottery_predictor_xgboost import LotteryPredictorXGBoost
from lottery_predictor_lightgbm import LotteryPredictorLightGBM
from lottery_predictor_catboost import LotteryPredictorCatBoost
from lottery_predictor_elasticnet import LotteryPredictorElasticNet
from lottery_predictor_adaboost import LotteryPredictorAdaBoost
from lottery_predictor_bagging import LotteryPredictorBagging
from lottery_predictor_bayesianridge import LotteryPredictorBayesianRidge
from lottery_predictor_svr_rbf import LotteryPredictorSVRRBF
from lottery_predictor_gradient_boosting import LotteryPredictorGradientBoosting
from lottery_predictor_rf_tuned import LotteryPredictorRFTuned
from lottery_predictor_dnn import LotteryPredictorDNN
from lottery_predictor_rf_variant import LotteryPredictorRFVariant
from lottery_predictor_column_based_linear_regression import LotteryPredictorColumnBasedLinearRegression
from lottery_predictor_column_based_svr import LotteryPredictorColumnBasedSVR
from lottery_predictor_lstm import LotteryPredictorLSTM
from lottery_predictor_lstm_column import LotteryPredictorLSTMColumn
from lottery_predictor_transformer import LotteryPredictorTransformer
from lottery_matcher import LotteryMatcher
from probability_calculator import ProbabilityCalculator
import pandas as pd
import numpy as np
import argparse

def calculate_distance(predictions, actual):
    return np.linalg.norm(predictions - actual)

# Función para manejar argumentos de línea de comandos
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process lottery predictions.')
    parser.add_argument('-r', type=str, choices=['Y', 'N'], default='Y',
                      help='Recalculate distances (Y) or load from file (N)')
    parser.add_argument('-nr', type=int, default=3,
                      help='Number of records to remove for prediction testing (default: 3)')
    return parser.parse_args()

# Función principal
def main():
    # Parsear argumentos de línea de comandos
    args = parse_arguments()

    # Ruta al archivo de datos
    file_path = 'historic_data/bal_results_i.txt'
    
    # Instanciar y cargar datos
    lottery_data = LotteryData(file_path)
    data = lottery_data.get_data()
    
    # Número de registros a eliminar desde los argumentos
    num_records_to_remove = args.nr

    # Nombre del archivo CSV para guardar/cargar las distancias
    distance_file = 'distances.csv'

    if args.r == 'Y':
        # Diccionario para almacenar las distancias
        distances = {
            'SVR': [],
            'Random Forest': [],
            'KNN': [],
            'Gradient Boosting': [],
            'XGBoost': [],
            'LightGBM': [],
            'CatBoost': [],
            'ElasticNet': [],
            'AdaBoost': [],
            'Bagging': [],
            'Bayesian Ridge': [],
            'SVR RBF': [],
            'Gradient Boosting Regressor': [],
            'Random Forest Tuned': [],
            'DNN': [],
            'RF Variant': [],
            'Column Based': [],
            'Column Based SVR': [],
            'LSTM': [],
            'LSTM Column': [],
            'Transformer': []
        }

        # Diccionario para almacenar el conteo de coincidencias
        matches = {
            'SVR': [],
            'Random Forest': [],
            'KNN': [],
            'Gradient Boosting': [],
            'XGBoost': [],
            'LightGBM': [],
            'CatBoost': [],
            'ElasticNet': [],
            'AdaBoost': [],
            'Bagging': [],
            'Bayesian Ridge': [],
            'SVR RBF': [],
            'Gradient Boosting Regressor': [],
            'Random Forest Tuned': [],
            'DNN': [],
            'RF Variant': [],
            'Column Based': [],
            'Column Based SVR': [],
            'LSTM': [],
            'LSTM Column': [],
            'Transformer': []
        }

        # Diccionario para almacenar los números coincidentes
        matching_numbers = {
            'Actual numbers': [],
            'SVR': [],
            'Random Forest': [],
            'KNN': [],
            'Gradient Boosting': [],
            'XGBoost': [],
            'LightGBM': [],
            'CatBoost': [],
            'ElasticNet': [],
            'AdaBoost': [],
            'Bagging': [],
            'Bayesian Ridge': [],
            'SVR RBF': [],
            'Gradient Boosting Regressor': [],
            'Random Forest Tuned': [],
            'DNN': [],
            'RF Variant': [],
            'Column Based': [],
            'Column Based SVR': [],
            'LSTM': [],
            'LSTM Column': [],
            'Transformer': [],
            'Count Found': [],
            'All Found Numbers': [],
            'All Found Complementary': []
        }

        predicted_numbers = {
            'Actual numbers': [],
            'SVR': [],
            'Random Forest': [],
            'KNN': [],
            'Gradient Boosting': [],
            'XGBoost': [],
            'LightGBM': [],
            'CatBoost': [],
            'ElasticNet': [],
            'AdaBoost': [],
            'Bagging': [],
            'Bayesian Ridge': [],
            'SVR RBF': [],
            'Gradient Boosting Regressor': [],
            'Random Forest Tuned': [],
            'DNN': [],
            'RF Variant': [],
            'Column Based': [],
            'Column Based SVR': [],
            'LSTM': [],
            'LSTM Column': [],
            'Transformer': [],
            'Count All Numbers Predicted': [],
            'Most Frequent Numbers': [],
            'Most Frequent Complementary': [],
            'Actual Numbers Found In Most Frequent': [],
            'Actual Complementary Found': []
        }

        # Iterate from 1 to num_records_to_remove (inclusive)
        for i in range(1, num_records_to_remove + 1):
            print(f"********************** Iteración con {i} registros eliminados **********************{data}")
            train_data = data[:-i]
            actual_values = data[-i:]

            # Function to handle predictions and matching
            def process_predictions(predictor_class, model_name):
                predictor = predictor_class(train_data)
                predictor.train()
                prediction = predictor.predict()
                
                # Si es el primer registro (i=1), no hay valores actuales para comparar
                if i == 1:
                    matcher = LotteryMatcher(prediction, [0, 0, 0, 0, 0, 0])  # Valores dummy para el matcher
                    distance = 0  # No calculamos distancia para la primera predicción
                    first_five_matches = 0
                    sixth_match = 0
                    matching_numbers_str = "To be predicted"
                else:
                    matcher = LotteryMatcher(prediction, actual_values.iloc[0])
                    distance = matcher.calculate_distance()
                    first_five_matches, sixth_match = matcher.calculate_matches()
                    matching_numbers_str = matcher.get_matching_numbers()

                distances[model_name].append(distance)
                matches[model_name].append(first_five_matches + sixth_match)
                matching_numbers[model_name].append(matching_numbers_str)
                predicted_numbers[model_name].append(prediction)

            # Process each model
            process_predictions(LotteryPredictorSVR, 'SVR')
            process_predictions(LotteryPredictorRF, 'Random Forest')
            process_predictions(LotteryPredictorKNN, 'KNN')
            process_predictions(LotteryPredictorGBR, 'Gradient Boosting')
            process_predictions(LotteryPredictorXGBoost, 'XGBoost')
            process_predictions(LotteryPredictorLightGBM, 'LightGBM')
            process_predictions(LotteryPredictorCatBoost, 'CatBoost')
            process_predictions(LotteryPredictorElasticNet, 'ElasticNet')
            process_predictions(LotteryPredictorAdaBoost, 'AdaBoost')
            process_predictions(LotteryPredictorBagging, 'Bagging')
            process_predictions(LotteryPredictorBayesianRidge, 'Bayesian Ridge')
            process_predictions(LotteryPredictorSVRRBF, 'SVR RBF')
            process_predictions(LotteryPredictorGradientBoosting, 'Gradient Boosting Regressor')
            process_predictions(LotteryPredictorRFTuned, 'Random Forest Tuned')
            process_predictions(LotteryPredictorDNN, 'DNN')
            process_predictions(LotteryPredictorRFVariant, 'RF Variant')
            process_predictions(LotteryPredictorColumnBasedLinearRegression, 'Column Based')
            process_predictions(LotteryPredictorColumnBasedSVR, 'Column Based SVR')
            process_predictions(LotteryPredictorLSTM, 'LSTM')
            process_predictions(LotteryPredictorLSTMColumn, 'LSTM Column')
            process_predictions(LotteryPredictorTransformer, 'Transformer')

            # Y modificar la parte donde se guardan los números actuales
            # Dentro del bucle for i in range(1, num_records_to_remove + 1):
            if i == 1:
                matching_numbers['Actual numbers'].append("To be predicted")
                predicted_numbers['Actual numbers'].append([0, 0, 0, 0, 0, 0])  # Valores dummy para el primer registro
            else:
                actual_numbers_str = ' - '.join(map(str, actual_values.iloc[0].tolist()))
                matching_numbers['Actual numbers'].append(actual_numbers_str)
                predicted_numbers['Actual numbers'].append(actual_values.iloc[0].tolist())

            # Calculate the count of unique numbers found for each algorithm for the current iteration
            unique_numbers = set()
            unique_complementary = set()
            for key in matching_numbers:
                if key != 'Actual numbers' and key != 'Count Found' and key != 'All Found Numbers' and key != 'All Found Complementary':
                    if i <= len(matching_numbers[key]):
                        numbers = matching_numbers[key][i-1].split('&')
                        # Separar números regulares y complementarios
                        for num in numbers:
                            num = num.strip()
                            if num:
                                # Asumiendo que el último número en cada grupo es el complementario
                                if len(unique_numbers) < 5:
                                    unique_numbers.add(num)
                                else:
                                    unique_complementary.add(num)

            # Append the count of unique numbers to 'Count Found'
            matching_numbers['Count Found'].append(len(unique_numbers) + len(unique_complementary))
            
            # Add all found numbers as arrays, separating regular and complementary numbers
            matching_numbers['All Found Numbers'].append(sorted(list(unique_numbers)))
            matching_numbers['All Found Complementary'].append(sorted(list(unique_complementary)))

            # Print the current state of matching_numbers for debugging
            print("matching_numbers", matching_numbers)

            # Calculate the count of unique numbers predicted for each algorithm for the current iteration
            unique_predicted_numbers = set()
            for key in predicted_numbers:
                if key != 'Count All Numbers Predicted':
                    # Check if the index i is within the bounds of the list
                    if i <= len(predicted_numbers[key]):
                        # Access the i-th element and split the string by '&'
                        numbers = '&'.join(map(str, predicted_numbers[key][i-1])).split('&')
                        unique_predicted_numbers.update(num.strip() for num in numbers if num.strip())

            # Append the count of unique numbers predicted across all algorithms to 'Count All Numbers Predicted'
            predicted_numbers['Count All Numbers Predicted'].append(len(unique_predicted_numbers))

            # Count frequency of each predicted number (separately for first 5 and complementary)
            number_frequency = {}
            complementary_frequency = {}
            for key in predicted_numbers:
                if key != 'Count All Numbers Predicted' and key != 'Most Frequent Numbers' and key != 'Most Frequent Complementary':
                    if i <= len(predicted_numbers[key]):
                        numbers = predicted_numbers[key][i-1]
                        # Contar frecuencias de los primeros 5 números
                        first_five = numbers[:5]
                        for num in first_five:
                            if num in number_frequency:
                                number_frequency[num] += 1
                            else:
                                number_frequency[num] = 1
                        
                        # Contar frecuencias del número complementario
                        complementary = numbers[5]
                        if complementary in complementary_frequency:
                            complementary_frequency[complementary] += 1
                        else:
                            complementary_frequency[complementary] = 1

            # Sort numbers by frequency in descending order
            sorted_numbers = sorted(number_frequency.items(), key=lambda x: x[1], reverse=True)
            sorted_complementary = sorted(complementary_frequency.items(), key=lambda x: x[1], reverse=True)
            
            print("\nFrecuencia de números predichos en la iteración", i)
            for number, frequency in sorted_numbers:
                print(f"Número {number}: {frequency} veces")
            
            print("\nFrecuencia de números complementarios en la iteración", i)
            for number, frequency in sorted_complementary:
                print(f"Complementario {number}: {frequency} veces")
            
            # Add the top 10 most frequent numbers and top 3 complementary numbers
            most_frequent_numbers = [num for num, _ in sorted_numbers[:10]]
            most_frequent_complementary = [num for num, _ in sorted_complementary[:3]]
            predicted_numbers['Most Frequent Numbers'].append(most_frequent_numbers)
            predicted_numbers['Most Frequent Complementary'].append(most_frequent_complementary)

            # Verificar coincidencias con los números actuales
            actual_first_five = actual_values.iloc[0].tolist()[:5]
            actual_complementary = actual_values.iloc[0].tolist()[5]
            
            # Encontrar coincidencias en los primeros 5 números
            matches_in_frequent = [num for num in actual_first_five if num in most_frequent_numbers]
            predicted_numbers['Actual Numbers Found In Most Frequent'].append(matches_in_frequent)
            
            # Encontrar coincidencia en el complementario
            complementary_match = [actual_complementary] if actual_complementary in most_frequent_complementary else []
            predicted_numbers['Actual Complementary Found'].append(complementary_match)

        # Asegurar que todas las listas de distancias tengan la misma longitud
        max_length = max(len(lst) for lst in distances.values())
        for key in distances:
            while len(distances[key]) < max_length:
                distances[key].append(float('inf'))  # Fill with a large value to indicate missing data

        # Asegurar que todas las listas de coincidencias tengan la misma longitud
        max_length_matches = max(len(lst) for lst in matches.values())
        for key in matches:
            while len(matches[key]) < max_length_matches:
                matches[key].append(0)  # Fill with zero to indicate no matches

        # Asegurar que todas las listas de números coincidentes tengan la misma longitud
        max_length_matching = max(len(lst) for lst in matching_numbers.values())
        for key in matching_numbers:
            while len(matching_numbers[key]) < max_length_matching:
                matching_numbers[key].append('')  # Fill with empty strings to indicate missing data

        # Crear un DataFrame para almacenar las distancias y calcular el promedio
        distance_df = pd.DataFrame(distances)
        distance_df.loc['Average'] = distance_df.mean()
        distance_df = distance_df.T  # Transpose the DataFrame

        # Mostrar la matriz de distancias con el promedio
        print("\nMatriz de distancias con promedio:")
        print(distance_df)

        # Guardar la matriz de distancias en un archivo CSV
        distance_df.to_csv(distance_file)

        # Crear un DataFrame para almacenar las coincidencias y calcular el promedio
        matches_df = pd.DataFrame(matches)
        matches_df.loc['Average'] = matches_df.mean()
        matches_df = matches_df.T  # Transpose the DataFrame

        # Mostrar la matriz de coincidencias con el promedio
        print("\nMatriz de coincidencias con promedio:")
        print(matches_df)

        # Guardar la matriz de coincidencias en un archivo CSV
        matches_df.to_csv('matches.csv')

        # Show the transposed matrix
        matching_numbers_df = pd.DataFrame(matching_numbers).T
        print("\nMatriz de números coincidentes:")
        print(matching_numbers_df)

        # Save the transposed matrix to a CSV file
        matching_numbers_df.to_csv('matching_numbers.csv')

        # Save the transposed matrix to a CSV file
        max_length_predicted = max(len(lst) for lst in predicted_numbers.values())
        for key in predicted_numbers:
            while len(predicted_numbers[key]) < max_length_predicted:
                predicted_numbers[key].append([])  # Fill with empty lists to indicate missing data
        predicted_numbers_df = pd.DataFrame(predicted_numbers).T
        print("\nMatriz de números predichos:")
        print(predicted_numbers_df)

        # Save the transposed matrix to a CSV file
        predicted_numbers_df.to_csv('predicted_numbers.csv')

        # Calcular y mostrar las probabilidades usando la nueva clase
        ProbabilityCalculator.print_probabilities(predicted_numbers_df)

    else:
        # Cargar la matriz de distancias desde el archivo CSV
        distance_df = pd.read_csv(distance_file, index_col=0)

        # Cargar la matriz de coincidencias desde el archivo CSV
        matches_df = pd.read_csv('matches.csv', index_col=0)

        # Cargar la matriz de números coincidentes desde el archivo CSV
        matching_numbers_df = pd.read_csv('transposed_matching_numbers.csv', index_col=0)

        # Cargar la matriz de números predichos desde el archivo CSV
        predicted_numbers_df = pd.read_csv('predicted_numbers.csv', index_col=0)

    # Mostrar la matriz de distancias con el promedio
    print("\nMatriz de distancias con promedio:")
    print(distance_df)

    # Mostrar la matriz de coincidencias con el promedio
    print("\nMatriz de coincidencias con promedio:")
    print(matches_df)

    # Mostrar la matriz de números coincidentes
    print("\nMatriz de números coincidentes:")
    print(matching_numbers_df)

    # Mostrar la matriz de números predichos
    print("\nMatriz de números predichos:")
    print(predicted_numbers_df)

    # Calcular y mostrar las probabilidades usando la nueva clase
    ProbabilityCalculator.print_probabilities(predicted_numbers_df)

if __name__ == "__main__":
    main() 