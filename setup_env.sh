#!/bin/bash

# Crear el entorno virtual si no existe
if [ ! -d "myenv" ]; then
  /usr/local/bin/python3 -m venv myenv
fi

# Activar el entorno virtual
source myenv/bin/activate

# Instalar pandas, tensorflow, keras-tuner y alpha_vantage dentro del entorno virtual
pip install pandas tensorflow scikit-learn keras-tuner alpha_vantage quandl yfinance matplotlib


# Informar al usuario sobre cómo activar el entorno virtual en futuras sesiones
echo "El entorno virtual ha sido configurado y los paquetes necesarios han sido instalados."
echo "Para activar el entorno virtual en el futuro, utiliza el comando:"
echo "source myenv/bin/activate"
