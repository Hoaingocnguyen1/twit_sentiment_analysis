#!/bin/bash
# Airflow Setup Script
# This script sets up Apache Airflow with PostgreSQL database

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Environment variables - cập nhật đường dẫn để phù hợp với local
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql://hoaingocnguyen:Hn20052004!@twitwarehouse.postgres.database.azure.com:5432/postgres?sslmode=require"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__PLUGINS_FOLDER="$AIRFLOW_HOME/plugins"
export AIRFLOW__CORE__BASE_LOG_FOLDER="$AIRFLOW_HOME/logs"
export AIRFLOW__WEBSERVER__SECRET_KEY="my_ultra_secret_key_123456"
export PYTHONPATH="$AIRFLOW_HOME:$(pwd)/src:$PYTHONPATH"

# Default values - thay đổi để phù hợp với local setup
AIRFLOW_HOME=${AIRFLOW_HOME:-$(pwd)/airflow-dev}  # Sử dụng folder airflow trong thư mục hiện tại
ADMIN_USERNAME=${ADMIN_USERNAME:-admin}
ADMIN_PASSWORD=${ADMIN_PASSWORD:-admin_password}
ADMIN_EMAIL=${ADMIN_EMAIL:-nn20052004@gmail.com}
ADMIN_FIRSTNAME=${ADMIN_FIRSTNAME:-FirstName}
ADMIN_LASTNAME=${ADMIN_LASTNAME:-LastName}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    if ! command_exists pip; then
        print_error "pip is not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p "$AIRFLOW_HOME"
    mkdir -p "$AIRFLOW_HOME/dags"
    mkdir -p "$AIRFLOW_HOME/plugins"
    mkdir -p "$AIRFLOW_HOME/logs"
    mkdir -p "$AIRFLOW_HOME/src"
    mkdir -p "$AIRFLOW_HOME/config"
    
    print_success "Directories created successfully"
}

# Function to initialize Airflow database
init_database() {
    print_status "Initializing Airflow database..."
    
    # Test database connection
    print_status "Testing database connection..."
    if ! airflow db check; then
        print_error "Database connection failed"
        print_error "Please check your database configuration"
        exit 1
    fi
    
    # Initialize database
    airflow db init
    
    print_success "Database initialized successfully"
}

# Function to create admin user
create_admin_user() {
    print_status "Creating admin user..."
    
    # Check if user already exists
    if airflow users list | grep -q "$ADMIN_USERNAME"; then
        print_warning "User '$ADMIN_USERNAME' already exists, skipping creation"
        return 0
    fi
    
    # Create admin user
    airflow users create \
        --username "$ADMIN_USERNAME" \
        --password "$ADMIN_PASSWORD" \
        --firstname "$ADMIN_FIRSTNAME" \
        --lastname "$ADMIN_LASTNAME" \
        --email "$ADMIN_EMAIL" \
        --role Admin
    
    print_success "Admin user created successfully"
    print_status "Username: $ADMIN_USERNAME"
    print_status "Password: $ADMIN_PASSWORD"
}

# Function to start Airflow webserver
start_webserver() {
    print_status "Starting Airflow webserver on port 8080..."
    
    # Start webserver in background
    nohup airflow webserver --port 8080 > "$AIRFLOW_HOME/logs/webserver.log" 2>&1 &
    WEBSERVER_PID=$!
    echo $WEBSERVER_PID > "$AIRFLOW_HOME/webserver.pid"
    
    print_success "Webserver started with PID: $WEBSERVER_PID"
    print_status "Webserver logs: $AIRFLOW_HOME/logs/webserver.log"
}

# Function to start Airflow scheduler
start_scheduler() {
    print_status "Starting Airflow scheduler..."
    
    # Start scheduler in background
    nohup airflow scheduler > "$AIRFLOW_HOME/logs/scheduler.log" 2>&1 &
    SCHEDULER_PID=$!
    echo $SCHEDULER_PID > "$AIRFLOW_HOME/scheduler.pid"
    
    print_success "Scheduler started with PID: $SCHEDULER_PID"
    print_status "Scheduler logs: $AIRFLOW_HOME/logs/scheduler.log"
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait for webserver to start
    sleep 10
    
    # Check webserver health
    if command_exists curl; then
        if curl -f http://localhost:8080/health >/dev/null 2>&1; then
            print_success "Webserver is healthy"
        else
            print_warning "Webserver health check failed"
        fi
    else
        print_warning "curl not found, skipping health check"
    fi
    
    # Check if scheduler is running
    if [ -f "$AIRFLOW_HOME/scheduler.pid" ]; then
        SCHEDULER_PID=$(cat "$AIRFLOW_HOME/scheduler.pid")
        if ps -p $SCHEDULER_PID > /dev/null; then
            print_success "Scheduler is running"
        else
            print_warning "Scheduler is not running"
        fi
    fi
}

# Function to stop services
stop_services() {
    print_status "Stopping Airflow services..."
    
    if [ -f "$AIRFLOW_HOME/webserver.pid" ]; then
        WEBSERVER_PID=$(cat "$AIRFLOW_HOME/webserver.pid")
        if ps -p $WEBSERVER_PID > /dev/null; then
            kill $WEBSERVER_PID
            print_success "Webserver stopped"
        fi
        rm -f "$AIRFLOW_HOME/webserver.pid"
    fi
    
    if [ -f "$AIRFLOW_HOME/scheduler.pid" ]; then
        SCHEDULER_PID=$(cat "$AIRFLOW_HOME/scheduler.pid")
        if ps -p $SCHEDULER_PID > /dev/null; then
            kill $SCHEDULER_PID
            print_success "Scheduler stopped"
        fi
        rm -f "$AIRFLOW_HOME/scheduler.pid"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Full setup (install, init, create user)"
    echo "  start     - Start webserver and scheduler"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Check service status"
    echo "  help      - Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  AIRFLOW_HOME      - Airflow home directory (default: ./airflow)"
    echo "  ADMIN_USERNAME    - Admin username (default: admin)"
    echo "  ADMIN_PASSWORD    - Admin password (default: admin_password)"
    echo "  ADMIN_EMAIL       - Admin email (default: nn20052004@gmail.com)"
    echo ""
    echo "Examples:"
    echo "  # Sử dụng folder airflow tùy chỉnh"
    echo "  AIRFLOW_HOME=/path/to/my/airflow $0 setup"
    echo ""
    echo "  # Setup với thông tin admin tùy chỉnh" 
    echo "  ADMIN_USERNAME=myuser ADMIN_PASSWORD=mypass $0 setup"
}

# Function to show service status
show_status() {
    print_status "Checking Airflow service status..."
    
    if [ -f "$AIRFLOW_HOME/webserver.pid" ]; then
        WEBSERVER_PID=$(cat "$AIRFLOW_HOME/webserver.pid")
        if ps -p $WEBSERVER_PID > /dev/null; then
            print_success "Webserver is running (PID: $WEBSERVER_PID)"
        else
            print_warning "Webserver is not running"
        fi
    else
        print_warning "Webserver PID file not found"
    fi
    
    if [ -f "$AIRFLOW_HOME/scheduler.pid" ]; then
        SCHEDULER_PID=$(cat "$AIRFLOW_HOME/scheduler.pid")
        if ps -p $SCHEDULER_PID > /dev/null; then
            print_success "Scheduler is running (PID: $SCHEDULER_PID)"
        else
            print_warning "Scheduler is not running"
        fi
    else
        print_warning "Scheduler PID file not found"
    fi
}

# Main function
main() {
    case "${1:-setup}" in
        setup)
            print_status "Starting Airflow setup..."
            check_prerequisites
            create_directories
            install_airflow
            init_database
            create_admin_user
            print_success "Airflow setup completed successfully!"
            print_status "Run '$0 start' to start the services"
            ;;
        start)
            start_webserver
            start_scheduler
            check_health
            print_success "Airflow services started successfully!"
            print_status "Webserver: http://localhost:8080"
            print_status "Username: $ADMIN_USERNAME"
            print_status "Password: $ADMIN_PASSWORD"
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            start_webserver
            start_scheduler
            check_health
            print_success "Airflow services restarted successfully!"
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Trap to ensure cleanup on script exit
trap 'echo "Script interrupted"; exit 130' INT

# Run main function with all arguments
main "$@"