#!/bin/bash

# 1. Iniciar el Backend en segundo plano (usando &)
# Ajusta 'backend/main.py' si tu archivo principal tiene otro nombre
python backend/main.py &

# 2. Iniciar el Frontend de Streamlit
# Usamos $PORT porque Render asigna un puerto dinámico
streamlit run frontend/main.py --server.port $PORT --server.address 0.0.0.0