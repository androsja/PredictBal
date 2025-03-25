import numpy as np
import pandas as pd

class LotteryMatcher:
    def __init__(self, predictions, actual):
        self.predictions = predictions
        self.actual = actual
        
        # Corregir el acceso a los datos usando iloc para Series
        if isinstance(actual, pd.Series):
            self.regular_predictions = predictions[:5]
            self.complementary_prediction = predictions[5]
            self.regular_actual = actual.iloc[:5].values
            self.complementary_actual = actual.iloc[5]
        else:
            self.regular_predictions = predictions[:5]
            self.complementary_prediction = predictions[5]
            self.regular_actual = actual[:5]
            self.complementary_actual = actual[5]

    def calculate_distance(self):
        """
        Calcula la distancia euclidiana, dando más peso al complementario
        """
        try:
            # Asegurar que los datos son arrays numpy
            regular_pred = np.array(self.regular_predictions, dtype=float)
            regular_act = np.array(self.regular_actual, dtype=float)
            
            # Distancia para números regulares
            regular_distance = np.linalg.norm(regular_pred - regular_act)
            
            # Distancia para el complementario (normalizada al rango 1-16)
            complementary_distance = abs(float(self.complementary_prediction) - float(self.complementary_actual)) / 16.0
            
            # Combinar distancias (70% regulares, 30% complementario)
            return 0.7 * regular_distance + 0.3 * complementary_distance
        except Exception as e:
            print(f"Error calculando distancia: {e}")
            return float('inf')

    def calculate_matches(self):
        """
        Calcula coincidencias separadas para números regulares y complementario
        Returns:
            tuple: (coincidencias_regulares, coincidencia_complementario)
        """
        try:
            # Verificar que hay suficientes datos
            if len(self.predictions) < 6 or len(self.actual) < 6:
                print("Datos insuficientes para comparar.")
                return 0, 0

            # Coincidencias en números regulares
            regular_matches = len(
                set(map(int, self.regular_predictions)) & 
                set(map(int, self.regular_actual))
            )
            
            # Coincidencia en complementario
            complementary_match = int(
                int(self.complementary_prediction) == int(self.complementary_actual)
            )
            
            return regular_matches, complementary_match
        except Exception as e:
            print(f"Error calculando coincidencias: {e}")
            return 0, 0

    def get_matching_numbers(self):
        """
        Obtiene los números coincidentes, separando regulares y complementario
        Returns:
            str: Números coincidentes formateados (regulares & complementario)
        """
        try:
            # Verificar que hay suficientes datos
            if len(self.predictions) < 6 or len(self.actual) < 6:
                return ""

            # Obtener coincidencias en números regulares
            regular_matches = sorted(
                set(map(int, self.regular_predictions)) & 
                set(map(int, self.regular_actual))
            )
            
            # Verificar coincidencia en complementario
            complementary_match = []
            if int(self.complementary_prediction) == int(self.complementary_actual):
                complementary_match = [int(self.complementary_prediction)]

            # Combinar resultados
            all_matches = []
            
            # Agregar coincidencias regulares
            if regular_matches:
                all_matches.extend([f"R{num}" for num in regular_matches])
            
            # Agregar coincidencia del complementario
            if complementary_match:
                all_matches.extend([f"C{num}" for num in complementary_match])
            
            return ' & '.join(map(str, all_matches))
        except Exception as e:
            print(f"Error obteniendo números coincidentes: {e}")
            return ""

    def get_detailed_matches(self):
        """
        Proporciona un informe detallado de las coincidencias
        Returns:
            dict: Diccionario con información detallada de coincidencias
        """
        try:
            regular_matches, complementary_match = self.calculate_matches()
            
            return {
                'regular_matches': regular_matches,
                'complementary_match': complementary_match,
                'total_matches': regular_matches + complementary_match,
                'regular_matching_numbers': sorted(
                    set(map(int, self.regular_predictions)) & 
                    set(map(int, self.regular_actual))
                ),
                'complementary_matched': int(self.complementary_prediction) == int(self.complementary_actual),
                'complementary_value': int(self.complementary_prediction) if complementary_match else None,
                'distance': self.calculate_distance()
            }
        except Exception as e:
            print(f"Error obteniendo detalles de coincidencias: {e}")
            return {
                'regular_matches': 0,
                'complementary_match': 0,
                'total_matches': 0,
                'regular_matching_numbers': [],
                'complementary_matched': False,
                'complementary_value': None,
                'distance': float('inf')
            }

    def print_match_summary(self):
        """
        Imprime un resumen de las coincidencias
        """
        try:
            details = self.get_detailed_matches()
            
            print("\nResumen de Coincidencias:")
            print(f"Números Regulares: {details['regular_matches']}/5")
            print(f"Complementario: {'Sí' if details['complementary_match'] else 'No'}")
            print(f"Total Coincidencias: {details['total_matches']}/6")
            print(f"Números Regulares Coincidentes: {details['regular_matching_numbers']}")
            if details['complementary_matched']:
                print(f"Complementario Coincidente: {details['complementary_value']}")
            print(f"Distancia: {details['distance']:.4f}")
        except Exception as e:
            print(f"Error imprimiendo resumen: {e}") 