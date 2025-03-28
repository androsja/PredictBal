import pandas as pd
import numpy as np
from itertools import combinations as iter_combinations
import logging

class LotteryAnalyzer:
    def __init__(self, predicted_numbers_df, matches_df, distance_df):
        """
        Inicializa el analizador con los DataFrames necesarios
        """
        self.predicted_numbers_df = predicted_numbers_df
        self.matches_df = matches_df
        self.distance_df = distance_df
        
        # Definir los rangos exactos para los números posibles
        self.regular_numbers = list(range(1, 44))     # [1, 2, ..., 43]
        self.complementary_numbers = list(range(1, 17))  # [1, 2, ..., 16]
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - ANÁLISIS: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def analyze(self, top_combinations=10):
        """
        Analiza los datos y retorna las mejores combinaciones junto con las matrices de probabilidad
        """
        self.logger.info("Iniciando análisis completo...")
        
        # Obtener matrices de probabilidad y probabilidades promedio
        probability_analysis = self._analyze_iteration_probabilities()
        
        # Generar combinaciones recomendadas basadas en las probabilidades promedio
        recommended_combinations = self._generate_recommended_combinations(
            {
                'regular': probability_analysis['regular_probabilities'],
                'complementary': probability_analysis['complementary_probabilities']
            },
            top_combinations
        )
        
        return {
            'recommended_combinations': recommended_combinations,
            'regular_matrix': probability_analysis['regular_matrix'],
            'complementary_matrix': probability_analysis['complementary_matrix'],
            'number_probabilities': {
                'regular': probability_analysis['regular_probabilities'],
                'complementary': probability_analysis['complementary_probabilities']
            }
        }

    def _analyze_iteration_probabilities(self):
        """
        Analiza las probabilidades de cada número por iteración y por algoritmo
        """
        self.logger.info("Analizando probabilidades por iteración y algoritmo...")
        
        # Crear matrices de probabilidades
        regular_matrix = pd.DataFrame(
            0, 
            index=self.regular_numbers,
            columns=self.predicted_numbers_df.columns
        )
        
        complementary_matrix = pd.DataFrame(
            0, 
            index=self.complementary_numbers,
            columns=self.predicted_numbers_df.columns
        )
        
        # Para cada iteración (columna)
        for col in self.predicted_numbers_df.columns:
            algorithms_count = 0
            
            # Para cada algoritmo
            for algorithm in self.predicted_numbers_df.index:
                if algorithm not in ['Actual numbers', 'Count All Numbers Predicted', 
                                   'Most Frequent Numbers', 'Most Frequent Complementary',
                                   'Actual Numbers Found In Most Frequent', 
                                   'Actual Complementary Found']:
                    
                    # Obtener predicción y convertir a lista si es necesario
                    pred = self.predicted_numbers_df.loc[algorithm][col]
                    if isinstance(pred, str):
                        # Si es string, convertir a lista
                        pred = eval(pred)
                    
                    if isinstance(pred, list) and len(pred) == 6:
                        algorithms_count += 1
                        
                        # Contar números regulares
                        for num in pred[:5]:
                            if isinstance(num, (int, float)) and 1 <= num <= 43:
                                regular_matrix.at[num, col] += 1
                        
                        # Contar complementario
                        comp_num = pred[5]
                        if isinstance(comp_num, (int, float)) and 1 <= comp_num <= 16:
                            complementary_matrix.at[comp_num, col] += 1
            
            # Normalizar las probabilidades para esta iteración
            if algorithms_count > 0:
                # Para números regulares: dividir por el total de predicciones posibles
                regular_matrix[col] = regular_matrix[col] / (algorithms_count * 5)
                
                # Para complementarios: dividir por el número de algoritmos
                complementary_matrix[col] = complementary_matrix[col] / algorithms_count
        
        # Calcular probabilidad promedio para cada número
        regular_probabilities = regular_matrix.mean(axis=1)
        complementary_probabilities = complementary_matrix.mean(axis=1)
        
        # Logging de resultados
        self.logger.info("\nProbabilidades promedio para números regulares:")
        for num, prob in regular_probabilities.nlargest(10).items():
            self.logger.info(f"Número {num:2d}: {prob:.4f}")
        
        self.logger.info("\nProbabilidades promedio para números complementarios:")
        for num, prob in complementary_probabilities.nlargest(5).items():
            self.logger.info(f"Número {num:2d}: {prob:.4f}")
        
        return {
            'regular_matrix': regular_matrix,
            'complementary_matrix': complementary_matrix,
            'regular_probabilities': regular_probabilities.to_dict(),
            'complementary_probabilities': complementary_probabilities.to_dict()
        }

    def _analyze_ignored_numbers(self):
        """
        Analiza números que los algoritmos tienden a ignorar pero que aparecen en resultados reales
        """
        self.logger.info("Analizando números ignorados por los algoritmos...")
        
        ignored_regular = {num: {'ignored_count': 0, 'actual_appearances': 0} for num in self.regular_numbers}
        ignored_complementary = {num: {'ignored_count': 0, 'actual_appearances': 0} for num in self.complementary_numbers}
        
        # Analizar cada iteración
        for col in self.predicted_numbers_df.columns[1:]:  # Excluir primera columna
            predicted_regular = set()
            predicted_complementary = set()
            
            # Recolectar números predichos
            for algorithm in self.predicted_numbers_df.index:
                if algorithm not in ['Actual numbers', 'Count All Numbers Predicted', 
                                   'Most Frequent Numbers', 'Most Frequent Complementary',
                                   'Actual Numbers Found In Most Frequent', 
                                   'Actual Complementary Found']:
                    pred = self.predicted_numbers_df.loc[algorithm][col]
                    if isinstance(pred, list) and len(pred) == 6:
                        predicted_regular.update(pred[:5])
                        predicted_complementary.add(pred[5])
            
            # Verificar números actuales
            actual = self.predicted_numbers_df.loc['Actual numbers'][col]
            if isinstance(actual, list) and len(actual) == 6:
                # Contar números ignorados que aparecieron
                for num in actual[:5]:
                    if num not in predicted_regular:
                        ignored_regular[num]['ignored_count'] += 1
                    ignored_regular[num]['actual_appearances'] += 1
                
                if actual[5] not in predicted_complementary:
                    ignored_complementary[actual[5]]['ignored_count'] += 1
                ignored_complementary[actual[5]]['actual_appearances'] += 1
        
        return {'regular': ignored_regular, 'complementary': ignored_complementary}

    def _calculate_final_probabilities(self, iteration_probs, ignored_analysis):
        """
        Calcula probabilidades finales considerando tanto predicciones como números ignorados
        """
        self.logger.info("Calculando probabilidades finales...")
        
        final_regular_probs = {}
        final_complementary_probs = {}
        
        # Factor de suavizado para números no predichos
        smoothing_factor = 0.01
        
        # Calcular para números regulares
        total_regular_prob = 0
        for num in self.regular_numbers:
            pred_prob = iteration_probs['regular'].get(num, 0)
            ignored_data = ignored_analysis['regular'][num]
            
            # Si el número nunca fue predicho pero apareció en resultados reales
            if pred_prob == 0 and ignored_data['actual_appearances'] > 0:
                pred_prob = smoothing_factor
            
            # Combinar probabilidades (70% predicciones, 30% factor ignorado)
            final_prob = max(pred_prob, smoothing_factor)  # Asegurar probabilidad mínima
            final_regular_probs[num] = final_prob
            total_regular_prob += final_prob
        
        # Normalizar probabilidades regulares
        if total_regular_prob > 0:
            for num in final_regular_probs:
                final_regular_probs[num] /= total_regular_prob
        
        # Calcular para números complementarios
        total_complementary_prob = 0
        for num in self.complementary_numbers:
            pred_prob = iteration_probs['complementary'].get(num, 0)
            ignored_data = ignored_analysis['complementary'][num]
            
            # Si el número nunca fue predicho pero apareció en resultados reales
            if pred_prob == 0 and ignored_data['actual_appearances'] > 0:
                pred_prob = smoothing_factor
            
            # Combinar probabilidades
            final_prob = max(pred_prob, smoothing_factor)  # Asegurar probabilidad mínima
            final_complementary_probs[num] = final_prob
            total_complementary_prob += final_prob
        
        # Normalizar probabilidades complementarias
        if total_complementary_prob > 0:
            for num in final_complementary_probs:
                final_complementary_probs[num] /= total_complementary_prob
        
        return {
            'regular': final_regular_probs,
            'complementary': final_complementary_probs
        }

    def _generate_recommended_combinations(self, probabilities, top_n):
        """
        Genera combinaciones basadas en las probabilidades calculadas
        """
        self.logger.info("Generando combinaciones recomendadas...")
        
        # Ordenar números por probabilidad
        sorted_regular = sorted(
            probabilities['regular'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        sorted_complementary = sorted(
            probabilities['complementary'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Tomar los números más probables
        top_regular = [num for num, _ in sorted_regular[:15]]  # Tomamos más para generar combinaciones
        top_complementary = sorted_complementary[:5]
        
        combinations_list = []
        for reg_combo in iter_combinations(top_regular, 5):
            for comp_num, comp_prob in top_complementary:
                regular_numbers = list(reg_combo)  # Convertimos la tupla a lista
                regular_prob = sum(probabilities['regular'][num] for num in regular_numbers)
                
                combinations_list.append({
                    'regular_numbers': sorted(regular_numbers),
                    'complementary': comp_num,
                    'confidence_score': (regular_prob + comp_prob) / 6,
                    'regular_probabilities': {num: probabilities['regular'][num] for num in regular_numbers},
                    'complementary_probability': comp_prob
                })
        
        # Ordenar por puntuación de confianza
        combinations_list.sort(key=lambda x: x['confidence_score'], reverse=True)
        return combinations_list[:top_n]

    def print_analysis_results(self, results):
        """
        Imprime solo las matrices de probabilidad
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
        
        print("\n=== MATRIZ DE PROBABILIDADES - NÚMEROS REGULARES (1-43) ===")
        print(results['regular_matrix'])
        
        print("\n=== MATRIZ DE PROBABILIDADES - NÚMEROS COMPLEMENTARIOS (1-16) ===")
        print(results['complementary_matrix']) 