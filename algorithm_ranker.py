import pandas as pd
import numpy as np

class AlgorithmRanker:
    def __init__(self, predicted_numbers_df, distance_df, matches_df):
        self.predicted_numbers_df = predicted_numbers_df
        self.distance_df = distance_df
        self.matches_df = matches_df
    
    def rank_algorithms(self):
        """
        Calcula y ordena los algoritmos según sus distancias promedio y coincidencias
        """
        algorithm_scores = {}
        
        for algorithm in self.distance_df.index:
            if algorithm != 'Average':
                # Obtener distancias y coincidencias, excluyendo primera iteración
                distances = self.distance_df.loc[algorithm][1:]
                matches = self.matches_df.loc[algorithm][1:]
                
                # Calcular métricas
                avg_distance = distances.mean()
                avg_matches = matches.mean()
                
                # Calcular score combinado (70% distancia, 30% coincidencias)
                distance_score = np.exp(-avg_distance)  # Convertir distancia a score (menor distancia = mayor score)
                match_score = avg_matches / 6  # Normalizar coincidencias (máximo 6 coincidencias posibles)
                
                final_score = (0.7 * distance_score) + (0.3 * match_score)
                
                algorithm_scores[algorithm] = {
                    'score': final_score,
                    'avg_distance': avg_distance,
                    'min_distance': distances.min(),
                    'max_distance': distances.max(),
                    'avg_matches': avg_matches
                }
        
        # Crear DataFrame con los resultados
        ranked_algorithms = pd.DataFrame.from_dict(algorithm_scores, orient='index')
        
        # Ordenar por score (mayor a menor)
        ranked_algorithms = ranked_algorithms.sort_values('score', ascending=False)
        
        return ranked_algorithms

    def print_ranking(self, ranked_algorithms):
        """
        Imprime el ranking de algoritmos de manera formateada
        """
        print("\n=== RANKING DE ALGORITMOS POR PRECISIÓN ===")
        print("-" * 100)
        print(f"{'Pos':^4} | {'Algoritmo':^30} | {'Score':^8} | {'Dist. Prom':^10} | {'Dist. Min':^8} | {'Dist. Max':^8} | {'Coincid.':^8}")
        print("-" * 100)
        
        for i, (algorithm, row) in enumerate(ranked_algorithms.iterrows(), 1):
            print(f"{i:^4} | {algorithm:<30} | {row['score']:8.4f} | {row['avg_distance']:10.4f} | "
                  f"{row['min_distance']:8.4f} | {row['max_distance']:8.4f} | {row['avg_matches']:8.2f}") 