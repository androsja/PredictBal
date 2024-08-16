#!/bin/bash

# Definir el archivo de entrada
input_file="historic_data/bal_results_i.txt"

# Archivos de salida
bal_five="historic_data/bal_five.txt"
bal_six="historic_data/bal_six.txt"
rev_five="historic_data/rev_five.txt"
rev_six="historic_data/rev_six.txt"

# Limpiar los archivos de salida si ya existen
> $bal_five
> $bal_six
> $rev_five
> $rev_six

# Leer el archivo línea por línea
line_number=0
while IFS= read -r line
do
    line_number=$((line_number + 1))
    
    # Separar los números de la línea utilizando " - " como delimitador
    IFS=' - ' read -r -a numbers <<< "$line"
    
    # Si la línea es impar
    if ((line_number % 2 != 0)); then
        # Extraer los primeros 5 números y el sexto número
        first_five="${numbers[0]} - ${numbers[1]} - ${numbers[2]} - ${numbers[3]} - ${numbers[4]}"
        sixth="${numbers[5]}"
        
        # Escribir en los archivos bal_five.txt y bal_six.txt
        echo "$first_five" >> $bal_five
        echo "$sixth" >> $bal_six
    else
        # Si la línea es par
        # Extraer los primeros 5 números y el sexto número
        first_five="${numbers[0]} - ${numbers[1]} - ${numbers[2]} - ${numbers[3]} - ${numbers[4]}"
        sixth="${numbers[5]}"
        
        # Escribir en los archivos rev_five.txt y rev_six.txt
        echo "$first_five" >> $rev_five
        echo "$sixth" >> $rev_six
    fi
done < "$input_file"

echo "Archivos generados exitosamente:"
echo "$bal_five"
echo "$bal_six"
echo "$rev_five"
echo "$rev_six"
