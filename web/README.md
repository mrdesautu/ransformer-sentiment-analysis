# 🌐 Interfaz Web - Transformer Sentiment Analysis

Una interfaz web interactiva y moderna para demostrar las capacidades del proyecto de análisis de sentimientos con transformers.

## ✨ Características

### 🎯 **Demo Interactivo**
- **Análisis individual**: Analiza texto en tiempo real
- **Análisis por lotes**: Procesa múltiples textos simultáneamente
- **Selección de modelo**: Cambia entre modelo pre-entrenado y fine-tuneado
- **Visualización de probabilidades**: Gráficos de distribución de confianza

### 📊 **Visualización de Métricas**
- **Curvas de entrenamiento**: Loss y accuracy por época
- **Métricas de rendimiento**: Accuracy, F1-score, Loss
- **Arquitectura del modelo**: Información detallada del transformer

### 🏗️ **Arquitectura del Sistema**
- **Diagrama interactivo**: Flujo de datos desde input hasta predicción
- **Stack tecnológico**: Tecnologías utilizadas en el proyecto
- **Información del proyecto**: Características y capacidades

## 🚀 Uso Rápido

### **Opción 1: Servidor Web Integrado**
```bash
# Desde el directorio raíz del proyecto
python serve_web.py

# Con opciones personalizadas
python serve_web.py --port 8080 --no-browser
```

### **Opción 2: Servidor Web Manual**
```bash
# Navegar al directorio web
cd web

# Servir con Python
python -m http.server 8080

# O con Node.js (si está instalado)
npx serve -p 8080
```

### **Opción 3: Usar con API**
```bash
# Terminal 1: Iniciar la API
python -m src.api --host 127.0.0.1 --port 8000

# Terminal 2: Iniciar la interfaz web
python serve_web.py --port 8080
```

## 🔧 Configuración

### **URLs y Endpoints**
- **Interfaz Web**: `http://localhost:8080`
- **API Backend**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### **Configuración de API**
La interfaz se conecta automáticamente a la API en `http://127.0.0.1:8000`. Para cambiar:

```javascript
// En web/app.js, línea 2
const API_BASE_URL = 'http://tu-servidor:puerto';
```

## 📱 Funcionalidades

### **1. Análisis de Texto Individual**
- Input: Textarea para ingreso de texto
- Output: Sentimiento detectado, confianza, gráfico de probabilidades
- Ejemplos: Botón para generar textos de prueba

### **2. Análisis por Lotes**
- Input: Múltiples textos (uno por línea)
- Output: Lista de resultados + gráfico de distribución
- Límite: 10 textos por lote (configurable)

### **3. Configuración del Modelo**
- Selector de modelo: Pre-entrenado vs Fine-tuneado
- Toggle de probabilidades: Mostrar/ocultar distribución
- Estado de API: Conectado/Desconectado/Cargando

### **4. Métricas y Visualización**
- Gráfico de entrenamiento: Loss y accuracy por época
- Círculos de rendimiento: Métricas clave animadas
- Información de arquitectura: Detalles del modelo

## 🎨 Diseño y UX

### **Características Visuales**
- **Diseño responsive**: Adaptable a móviles y tablets
- **Tema moderno**: Gradientes, sombras y animaciones
- **Tipografía**: Inter font para legibilidad
- **Iconos**: Font Awesome para iconografía consistente

### **Interactividad**
- **Navegación suave**: Scroll automático entre secciones
- **Estados de carga**: Spinners y overlays
- **Feedback visual**: Colores para sentimientos positivos/negativos
- **Animaciones**: Transiciones suaves en hover y click

### **Accesibilidad**
- **Contraste adecuado**: Cumple estándares WCAG
- **Navegación por teclado**: Enter para enviar, Tab para navegar
- **Mensajes descriptivos**: Estados de error claros
- **Responsive design**: Funciona en todos los dispositivos

## 🔗 Integración con Backend

### **Endpoints Utilizados**
```javascript
// Health check
GET /health

// Modelo info
GET /model/info

// Predicción individual
POST /predict
POST /predict/probabilities

// Predicción por lotes
POST /predict/batch
```

### **Manejo de Errores**
- **API desconectada**: Modo demo con datos simulados
- **Errores de red**: Mensajes informativos al usuario
- **Timeout**: Reintentos automáticos
- **Validación**: Verificación de input en frontend

## 📊 Datos de Demo

Cuando la API no está disponible, la interfaz usa datos simulados:

```javascript
// Análisis basado en palabras clave
const positiveWords = ['good', 'great', 'excellent', 'amazing', 'love'];
const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'horrible'];

// Confianza simulada basada en coincidencias
confidence = 0.7 + (matches * 0.1);
```

## 🛠️ Tecnologías

### **Frontend**
- **HTML5**: Estructura semántica
- **CSS3**: Flexbox, Grid, animaciones
- **JavaScript ES6+**: Async/await, fetch API
- **Chart.js**: Gráficos interactivos
- **Font Awesome**: Iconografía

### **Backend Integration**
- **Fetch API**: Comunicación con FastAPI
- **JSON**: Intercambio de datos
- **CORS**: Configuración cross-origin
- **Error Handling**: Manejo robusto de errores

## 🔧 Personalización

### **Colores y Tema**
```css
/* Variables principales en styles.css */
--primary-color: #667eea;
--secondary-color: #764ba2;
--success-color: #28a745;
--danger-color: #dc3545;
```

### **Configuración de API**
```javascript
// Configuración en app.js
const API_BASE_URL = 'http://127.0.0.1:8000';
const POLLING_INTERVAL = 5000; // ms
```

### **Textos de Ejemplo**
```javascript
// Personalizar ejemplos en app.js
const exampleTexts = [
    "Tu texto de ejemplo aquí",
    "Otro ejemplo personalizado"
];
```

## 📱 Responsive Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px  
- **Desktop**: > 1024px

Adaptaciones automáticas:
- Navegación collapse en móvil
- Grid responsive para métricas
- Arquitectura vertical en pantallas pequeñas

## 🚀 Deployment

### **Servidor Web Local**
```bash
# Desarrollo
python serve_web.py --port 8080

# Producción simple
python -m http.server 8080 --directory web
```

### **Servidor Web Avanzado**
```bash
# Con nginx (ejemplo de configuración)
server {
    listen 80;
    root /path/to/transformer/web;
    index index.html;
    
    location /api/ {
        proxy_pass http://localhost:8000/;
    }
}
```

### **Docker**
```dockerfile
FROM nginx:alpine
COPY web /usr/share/nginx/html
EXPOSE 80
```

## 🔍 Testing

### **Tests Manuales**
1. ✅ Conexión a API: Verificar estado en header
2. ✅ Análisis individual: Probar con textos positivos/negativos
3. ✅ Análisis por lotes: Múltiples textos simultáneos
4. ✅ Responsive: Redimensionar ventana
5. ✅ Navegación: Links y scroll suave

### **Tests Automatizados** (Futuro)
```javascript
// Ejemplo con Jest/Cypress
describe('Sentiment Analysis Interface', () => {
    it('should analyze text and show results', () => {
        cy.visit('http://localhost:8080');
        cy.get('#text-input').type('Great movie!');
        cy.get('#analyze-btn').click();
        cy.get('#single-result').should('be.visible');
    });
});
```

## 📈 Métricas de Uso

La interfaz registra (localmente):
- Textos analizados
- Tiempo de respuesta
- Errores de API
- Patrones de uso

## 🎯 Próximas Mejoras

- [ ] **Authentication**: Login y perfiles de usuario
- [ ] **History**: Historial de análisis
- [ ] **Export**: Descargar resultados en CSV/JSON
- [ ] **Themes**: Modo oscuro/claro
- [ ] **Real-time**: WebSocket para análisis en vivo
- [ ] **Mobile App**: PWA o React Native
- [ ] **Analytics**: Google Analytics integration
- [ ] **A/B Testing**: Comparar diferentes modelos

## 🆘 Troubleshooting

### **Problemas Comunes**

**Q: La API no se conecta**
```bash
# Verificar que la API esté corriendo
curl http://localhost:8000/health

# Revisar CORS en app.js
# Verificar puertos correctos
```

**Q: Los gráficos no se muestran**
```bash
# Verificar Chart.js en consola del navegador
# Comprobar dimensiones de canvas
# Revisar datos en console.log
```

**Q: Estilos no se cargan**
```bash
# Verificar ruta de styles.css
# Comprobar servidor web corriendo
# Revisar permisos de archivos
```

**Q: JavaScript no funciona**
```bash
# Abrir DevTools (F12)
# Revisar errores en Console
# Verificar que app.js se carga correctamente
```

---

## 🎉 ¡Disfruta de la Demo!

La interfaz está diseñada para mostrar de forma atractiva y profesional las capacidades del proyecto de análisis de sentimientos con transformers. 

**¿Preguntas o mejoras?** ¡Experimenta con el código y personaliza según tus necesidades!