import pandas as pd
from collections import Counter
import random

class BestPredictor:
    def __init__(self, predicted_numbers_df, distance_df, matches_df):
        self.predicted_numbers_df = predicted_numbers_df
        self.distance_df = distance_df
        self.matches_df = matches_df

    def get_recommendations(self, top_n=3):
        """
        Obtiene una lista de recomendaciones de números de lotería,
        donde la cantidad de recomendaciones está determinada por top_n.

        Args:
            top_n (int): El número de algoritmos principales a considerar para las recomendaciones.

        Returns:
            list: Una lista de listas, donde cada lista interna es una recomendación de 6 números.
        """
        # --- Step 1: Identify the best performing algorithms based on the average distance ---
        sorted_distance_df = self.distance_df.sort_values(by='Average')
        top_algorithms = sorted_distance_df.index[:top_n].tolist()
        print(f"Algoritmos top {top_n}: {top_algorithms}")

        all_recommendations = []
        for algorithm in top_algorithms:
            if algorithm in self.predicted_numbers_df.index:
                prediction = self.predicted_numbers_df.loc[algorithm].iloc[0] # Get the prediction for the algorithm

                try:
                    # Assuming the prediction is a string representation of a list
                    numbers = [int(n.strip()) for n in prediction.strip('[]').split(',')]
                except AttributeError:
                    try:
                        numbers = [int(n) for n in prediction.strip('[]').split(',')]
                    except:
                        numbers = []

                if len(numbers) == 6:
                    first_five = sorted(list(set([num for num in numbers[:5] if 1 <= num <= 43])))
                    complement = numbers[5] if 1 <= numbers[5] <= 16 else random.randint(1, 16)

                    # Ensure we have exactly 5 unique numbers for the first five
                    if len(first_five) == 5:
                        all_recommendations.append(first_five + [complement])
                    else:
                        # Fallback: Generate a random recommendation if the prediction is not valid
                        first_five_random = sorted(random.sample(range(1, 44), 5))
                        complement_random = random.randint(1, 16)
                        all_recommendations.append(first_five_random + [complement_random])
                else:
                    # Fallback for incorrect number of predictions
                    first_five_random = sorted(random.sample(range(1, 44), 5))
                    complement_random = random.randint(1, 16)
                    all_recommendations.append(first_five_random + [complement_random])
            else:
                print(f"Advertencia: No se encontraron predicciones para el algoritmo {algorithm}")

        return all_recommendations