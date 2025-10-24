#!/bin/bash

# 🚀 Script de Deployment Completo para Transformer Web Interface
# Autor: AI Assistant
# Versión: 1.0

set -e  # Salir en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración por defecto
PROJECT_NAME="transformer-sentiment"
WEB_PORT=8080
API_PORT=8000
PYTHON_ENV="venv"
BROWSER_OPEN=true
KILL_EXISTING=true

# Funciones de utilidad
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                🤖 TRANSFORMER WEB DEPLOYMENT 🌐                 ║"
    echo "║                                                                  ║"
    echo "║  Desplegando interfaz web completa para análisis de sentimientos ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_help() {
    echo "Uso: $0 [OPCIONES]"
    echo ""
    echo "Opciones:"
    echo "  -w, --web-port PORT     Puerto para la interfaz web (default: 8080)"
    echo "  -a, --api-port PORT     Puerto para la API (default: 8000)"
    echo "  -e, --env ENV_NAME      Nombre del entorno virtual (default: venv)"
    echo "  --no-browser           No abrir browser automáticamente"
    echo "  --no-kill              No matar procesos existentes"
    echo "  --api-only             Solo iniciar API"
    echo "  --web-only             Solo iniciar interfaz web"
    echo "  --full                 Deployment completo (API + Web + Tests)"
    echo "  --docker               Usar Docker para deployment"
    echo "  --production           Configuración de producción"
    echo "  -h, --help             Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0                     # Deployment estándar"
    echo "  $0 --full             # Deployment completo con tests"
    echo "  $0 --web-only -w 3000 # Solo web en puerto 3000"
    echo "  $0 --production       # Deployment de producción"
}

check_dependencies() {
    log_info "Verificando dependencias..."
    
    # Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 no está instalado"
        exit 1
    fi
    
    # pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 no está instalado"
        exit 1
    fi
    
    log_success "Dependencias básicas verificadas"
}

check_ports() {
    log_info "Verificando disponibilidad de puertos..."
    
    if lsof -Pi :$WEB_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        if [ "$KILL_EXISTING" = true ]; then
            log_warning "Puerto $WEB_PORT ocupado. Matando proceso..."
            lsof -ti:$WEB_PORT | xargs kill -9 2>/dev/null || true
        else
            log_error "Puerto $WEB_PORT ya está en uso"
            exit 1
        fi
    fi
    
    if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        if [ "$KILL_EXISTING" = true ]; then
            log_warning "Puerto $API_PORT ocupado. Matando proceso..."
            lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
        else
            log_error "Puerto $API_PORT ya está en uso"
            exit 1
        fi
    fi
    
    log_success "Puertos disponibles"
}

setup_environment() {
    log_info "Configurando entorno Python..."
    
    # Activar entorno virtual si existe
    if [ -d "$PYTHON_ENV" ]; then
        source $PYTHON_ENV/bin/activate
        log_success "Entorno virtual activado: $PYTHON_ENV"
    else
        log_warning "Entorno virtual no encontrado: $PYTHON_ENV"
        log_info "Creando nuevo entorno virtual..."
        python3 -m venv $PYTHON_ENV
        source $PYTHON_ENV/bin/activate
        log_success "Nuevo entorno virtual creado y activado"
    fi
    
    # Instalar/actualizar dependencias
    if [ -f "requirements.txt" ]; then
        log_info "Instalando dependencias..."
        pip install -r requirements.txt
        log_success "Dependencias instaladas"
    else
        log_warning "requirements.txt no encontrado"
    fi
}

start_api() {
    log_info "Iniciando API en puerto $API_PORT..."
    
    # Verificar que el módulo API existe
    if [ ! -f "src/api.py" ]; then
        log_error "API no encontrada en src/api.py"
        return 1
    fi
    
    # Iniciar API en background
    nohup python -m src.api --host 127.0.0.1 --port $API_PORT > api.log 2>&1 &
    API_PID=$!
    echo $API_PID > api.pid
    
    # Esperar a que la API esté lista
    log_info "Esperando a que la API esté lista..."
    for i in {1..30}; do
        if curl -s http://127.0.0.1:$API_PORT/health > /dev/null 2>&1; then
            log_success "API iniciada correctamente (PID: $API_PID)"
            return 0
        fi
        sleep 1
    done
    
    log_error "La API no pudo iniciarse en 30 segundos"
    return 1
}

start_web() {
    log_info "Iniciando interfaz web en puerto $WEB_PORT..."
    
    # Verificar que los archivos web existen
    if [ ! -f "web/index.html" ]; then
        log_error "Interfaz web no encontrada en web/index.html"
        return 1
    fi
    
    # Hacer ejecutable el servidor si no lo es
    if [ -f "serve_web.py" ]; then
        chmod +x serve_web.py
        
        # Iniciar servidor web personalizado
        if [ "$BROWSER_OPEN" = true ]; then
            nohup python serve_web.py --port $WEB_PORT > web.log 2>&1 &
        else
            nohup python serve_web.py --port $WEB_PORT --no-browser > web.log 2>&1 &
        fi
    else
        # Usar servidor HTTP básico de Python
        cd web
        if [ "$BROWSER_OPEN" = true ]; then
            nohup python -m http.server $WEB_PORT > ../web.log 2>&1 &
            open http://localhost:$WEB_PORT 2>/dev/null || true
        else
            nohup python -m http.server $WEB_PORT > ../web.log 2>&1 &
        fi
        cd ..
    fi
    
    WEB_PID=$!
    echo $WEB_PID > web.pid
    
    # Verificar que el servidor web está funcionando
    sleep 2
    if curl -s http://localhost:$WEB_PORT > /dev/null 2>&1; then
        log_success "Interfaz web iniciada correctamente (PID: $WEB_PID)"
        return 0
    else
        log_error "La interfaz web no pudo iniciarse"
        return 1
    fi
}

run_tests() {
    log_info "Ejecutando tests del proyecto..."
    
    # Tests de API
    if [ -d "tests" ]; then
        python -m pytest tests/ -v
    else
        log_warning "Directorio de tests no encontrado"
    fi
    
    # Test de health check
    if curl -s http://127.0.0.1:$API_PORT/health | grep -q "healthy"; then
        log_success "API health check: ✅ PASS"
    else
        log_error "API health check: ❌ FAIL"
    fi
    
    # Test de interfaz web
    if curl -s http://localhost:$WEB_PORT | grep -q "Transformer"; then
        log_success "Web interface check: ✅ PASS"
    else
        log_error "Web interface check: ❌ FAIL"
    fi
}

show_status() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    🎉 DEPLOYMENT COMPLETADO 🎉                  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📊 Estado de servicios:${NC}"
    
    # Verificar API
    if curl -s http://127.0.0.1:$API_PORT/health > /dev/null 2>&1; then
        echo -e "  🟢 API: ${GREEN}RUNNING${NC} en http://127.0.0.1:$API_PORT"
        echo -e "     📚 Docs: http://127.0.0.1:$API_PORT/docs"
    else
        echo -e "  🔴 API: ${RED}DOWN${NC}"
    fi
    
    # Verificar Web
    if curl -s http://localhost:$WEB_PORT > /dev/null 2>&1; then
        echo -e "  🟢 Web: ${GREEN}RUNNING${NC} en http://localhost:$WEB_PORT"
    else
        echo -e "  🔴 Web: ${RED}DOWN${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}🔧 Comandos útiles:${NC}"
    echo -e "  ${YELLOW}Ver logs API:${NC}     tail -f api.log"
    echo -e "  ${YELLOW}Ver logs Web:${NC}     tail -f web.log"
    echo -e "  ${YELLOW}Parar servicios:${NC}  $0 --stop"
    echo -e "  ${YELLOW}Reiniciar:${NC}        $0 --restart"
    echo ""
    
    if [ "$BROWSER_OPEN" = true ]; then
        echo -e "${GREEN}🌐 Abriendo navegador...${NC}"
        if command -v open &> /dev/null; then
            open http://localhost:$WEB_PORT
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:$WEB_PORT
        fi
    fi
}

stop_services() {
    log_info "Deteniendo servicios..."
    
    # Parar API
    if [ -f "api.pid" ]; then
        API_PID=$(cat api.pid)
        kill $API_PID 2>/dev/null || true
        rm api.pid
        log_success "API detenida"
    fi
    
    # Parar Web
    if [ -f "web.pid" ]; then
        WEB_PID=$(cat web.pid)
        kill $WEB_PID 2>/dev/null || true
        rm web.pid
        log_success "Interfaz web detenida"
    fi
    
    # Limpiar puertos por si acaso
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$WEB_PORT | xargs kill -9 2>/dev/null || true
}

create_production_config() {
    log_info "Creando configuración de producción..."
    
    # Nginx config
    cat > nginx.conf << EOF
server {
    listen 80;
    server_name localhost;
    
    # Interfaz web
    location / {
        root $(pwd)/web;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }
    
    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:$API_PORT/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOF
    
    # Docker compose para producción
    cat > docker-compose.prod.yml << EOF
version: '3.8'
services:
  api:
    build: .
    ports:
      - "$API_PORT:$API_PORT"
    environment:
      - ENV=production
    restart: unless-stopped
    
  web:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./web:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - api
    restart: unless-stopped
EOF
    
    log_success "Configuración de producción creada"
}

docker_deployment() {
    log_info "Iniciando deployment con Docker..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker no está instalado"
        exit 1
    fi
    
    # Build imagen
    docker build -t $PROJECT_NAME .
    
    # Run con docker-compose
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d
        log_success "Servicios iniciados con Docker"
    else
        log_error "docker-compose.yml no encontrado"
        exit 1
    fi
}

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--web-port)
            WEB_PORT="$2"
            shift 2
            ;;
        -a|--api-port)
            API_PORT="$2"
            shift 2
            ;;
        -e|--env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        --no-browser)
            BROWSER_OPEN=false
            shift
            ;;
        --no-kill)
            KILL_EXISTING=false
            shift
            ;;
        --api-only)
            MODE="api-only"
            shift
            ;;
        --web-only)
            MODE="web-only"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --docker)
            MODE="docker"
            shift
            ;;
        --production)
            MODE="production"
            shift
            ;;
        --stop)
            stop_services
            exit 0
            ;;
        --restart)
            stop_services
            sleep 2
            # Continuar con el deployment normal
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Banner de inicio
print_banner

# Verificaciones iniciales
check_dependencies
check_ports

# Deployment según modo
case ${MODE:-"standard"} in
    "api-only")
        setup_environment
        start_api
        ;;
    "web-only")
        start_web
        ;;
    "docker")
        docker_deployment
        ;;
    "production")
        create_production_config
        setup_environment
        start_api
        start_web
        ;;
    "full")
        setup_environment
        start_api
        start_web
        run_tests
        ;;
    *)
        setup_environment
        start_api
        start_web
        ;;
esac

# Mostrar estado final
show_status

# Cleanup en exit
trap 'log_info "Limpiando..."; stop_services' EXIT

# Mantener script corriendo
log_info "Presiona Ctrl+C para detener todos los servicios..."
while true; do
    sleep 10
    
    # Verificar que los servicios siguen corriendo
    if [ "${MODE:-"standard"}" != "web-only" ]; then
        if ! curl -s http://127.0.0.1:$API_PORT/health > /dev/null 2>&1; then
            log_error "API caída. Reiniciando..."
            start_api
        fi
    fi
    
    if [ "${MODE:-"standard"}" != "api-only" ]; then
        if ! curl -s http://localhost:$WEB_PORT > /dev/null 2>&1; then
            log_error "Interfaz web caída. Reiniciando..."
            start_web
        fi
    fi
done