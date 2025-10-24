# Interpretabilidad del Modelo

## Funcionalidades Agregadas

### 1. Visualización de Atención
- **Resumen de Atención**: Muestra cómo se distribuye la atención across capas y cabezas
- **Mapa de Calor**: Visualización detallada de la atención entre tokens
- **Visualización Interactiva**: Permite explorar diferentes capas y cabezas de atención

### 2. Análisis SHAP (Opcional)
- Explicaciones basadas en valores SHAP
- Requiere instalación: `pip install shap`

### 3. Importancia de Tokens
- Muestra qué tokens reciben más atención
- Barras interactivas con puntuaciones de importancia

## Endpoints de API

### `/interpret` (POST)
Análisis completo de interpretabilidad
```json
{
  "text": "Texto a analizar"
}
```

### `/interpret/attention` (POST)
Datos detallados de atención para visualización interactiva
```json
{
  "text": "Texto a analizar"
}
```

## Interfaz Web

### Nueva Sección: Interpretabilidad
- Accesible desde la navegación principal
- Tabs para diferentes visualizaciones:
  - **Resumen**: Gráficos generales de atención
  - **Mapa de Calor**: Visualización detallada
  - **Interactivo**: Exploración de capas/cabezas

### Controles Interactivos
- Selector de capa
- Selector de cabeza de atención
- Visualización en tiempo real

## Uso

1. Ingresa un texto en la sección de Interpretabilidad
2. Haz clic en "Analizar Interpretabilidad"
3. Explora las diferentes visualizaciones usando los tabs
4. Usa los controles interactivos para examinar capas específicas

## Dependencias Opcionales

Para funcionalidad completa, instala:
```bash
pip install shap
```

## Archivos Modificados

- `src/api.py`: Nuevos endpoints de interpretabilidad
- `src/interpretability.py`: Módulo de interpretabilidad (ya existía)
- `web/index.html`: Nueva sección de interpretabilidad
- `web/styles.css`: Estilos para visualizaciones
- `web/app.js`: JavaScript para interactividad

## Notas Técnicas

- Las visualizaciones se generan en el servidor usando matplotlib
- Las imágenes se envían como base64 al frontend
- El backend maneja automáticamente modelos sin SHAP disponible
- Responsive design para dispositivos móviles