print("=== DIAGNÓSTICO DE IMPORTS ===")
import sys
print("Python path:")
for p in sys.path:
    print(" ", p)

print("\n=== VERIFICANDO LIBRERÍAS ===")
try:
    import seaborn as sns
    print("✓ seaborn OK - Versión:", sns.__version__)
except ImportError as e:
    print("✗ seaborn FALLÓ:", e)

try:
    from core.analyzer import NouxTrendsAnalyzer
    print("✓ NouxTrendsAnalyzer OK")
except ImportError as e:
    print("✗ NouxTrendsAnalyzer FALLÓ:", e)

print("\n=== DIAGNÓSTICO COMPLETADO ===")