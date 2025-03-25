class ProbabilityCalculator:
    @staticmethod
    def calculate_match_probabilities(df):
        """
        Calcula las probabilidades de coincidencias para diferentes escenarios.
        
        Args:
            df (pandas.DataFrame): DataFrame con los números predichos y actuales
            
        Returns:
            tuple: (prob_6_matches, prob_5_matches, prob_4_matches, prob_3_matches)
                - prob_6_matches: Probabilidad de tener 5 números regulares + 1 complementario
                - prob_5_matches: Probabilidad de tener 5 números en total
                - prob_4_matches: Probabilidad de tener 4 números en total
                - prob_3_matches: Probabilidad de tener 3 números en total
        """
        total_iterations = len(df.loc['Actual Numbers Found In Most Frequent'])
        if total_iterations == 0:
            return 0, 0, 0, 0

        count_6_matches = 0
        count_5_matches = 0
        count_4_matches = 0
        count_3_matches = 0

        for i in range(total_iterations):
            regular_matches = len(df.loc['Actual Numbers Found In Most Frequent'][i])
            complementary_matches = len(df.loc['Actual Complementary Found'][i])
            total_matches = regular_matches + complementary_matches

            if total_matches >= 6:
                count_6_matches += 1
            if total_matches >= 5:
                count_5_matches += 1
            if total_matches >= 4:
                count_4_matches += 1
            if total_matches >= 3:
                count_3_matches += 1

        prob_6_matches = (count_6_matches / total_iterations) * 100
        prob_5_matches = (count_5_matches / total_iterations) * 100
        prob_4_matches = (count_4_matches / total_iterations) * 100
        prob_3_matches = (count_3_matches / total_iterations) * 100

        return prob_6_matches, prob_5_matches, prob_4_matches, prob_3_matches

    @staticmethod
    def print_probabilities(df):
        """
        Calcula y muestra las probabilidades de coincidencias.
        
        Args:
            df (pandas.DataFrame): DataFrame con los números predichos y actuales
        """
        prob_6, prob_5, prob_4, prob_3 = ProbabilityCalculator.calculate_match_probabilities(df)
        print("\nProbabilidades de coincidencias:")
        print(f"Probabilidad de 6 números coincidentes (5 regulares + 1 complementario): {prob_6:.2f}%")
        print(f"Probabilidad de 5 números coincidentes: {prob_5:.2f}%")
        print(f"Probabilidad de 4 números coincidentes: {prob_4:.2f}%")
        print(f"Probabilidad de 3 números coincidentes: {prob_3:.2f}%") 