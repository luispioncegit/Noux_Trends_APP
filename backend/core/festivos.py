import pandas as pd
from datetime import datetime, date

class EcuadorHolidays:
    """Calendario de días festivos Ecuador 2023"""
    
    @staticmethod
    def get_holidays_2023():
        """Retorna lista de días festivos 2023 en Ecuador"""
        return [
            # Festivos fijos
            date(2023, 1, 1),   # Año Nuevo
            date(2023, 1, 6),   # Día de los Reyes Magos
            date(2023, 2, 20),  # Carnaval
            date(2023, 2, 21),  # Carnaval
            date(2023, 4, 7),   # Viernes Santo
            date(2023, 5, 1),   # Día del Trabajo
            date(2023, 5, 24),  # Batalla de Pichincha
            date(2023, 8, 10),  # Primer Grito de Independencia
            date(2023, 10, 9),  # Independencia de Guayaquil
            date(2023, 11, 2),  # Día de los Difuntos
            date(2023, 11, 3),  # Independencia de Cuenca
            date(2023, 12, 25), # Navidad
            date(2023, 12, 31), # Fin de Año
            
            # Festivos movibles (aproximados para 2023)
            date(2023, 2, 27),  # Lunes de Carnaval
            date(2023, 5, 26),  # Corpus Christi
        ]
    
    @staticmethod
    def is_holiday(fecha):
        """Verifica si una fecha es festivo en Ecuador"""
        holidays_2023 = EcuadorHolidays.get_holidays_2023()
        return fecha.date() in holidays_2023
    
    @staticmethod
    def is_weekend(fecha):
        """Verifica si es fin de semana"""
        return fecha.weekday() >= 5  # 5 = Saturday, 6 = Sunday
    
    @staticmethod
    def get_day_type(fecha):
        """Clasifica el tipo de día"""
        if EcuadorHolidays.is_holiday(fecha):
            return "festivo"
        elif EcuadorHolidays.is_weekend(fecha):
            return "fin_semana"
        else:
            return "laboral"