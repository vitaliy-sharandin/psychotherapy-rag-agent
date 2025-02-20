# Monitoring Setup

This project sets up a local deployment of Prometheus and Grafana for monitoring purposes.

## Prerequisites

- Docker
- Docker Compose

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd monitoring
   ```

2. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Access Prometheus at `http://localhost:9090`.

4. Access Grafana at `http://localhost:3000`.
   - Default login: `admin/admin`

## Configuration Files

- **Prometheus Configuration**: Located in `prometheus/prometheus.yml`
- **Alerting Rules**: Located in `prometheus/alert-rules.yml`
- **Grafana Dashboard Provisioning**: Located in `grafana/provisioning/dashboards/dashboard.yml`
- **Grafana Data Source Configuration**: Located in `grafana/provisioning/datasources/datasource.yml`
- **Main Dashboard**: Located in `grafana/dashboards/main-dashboard.json`

## Monitoring

This setup allows you to monitor your applications and infrastructure using Prometheus and visualize the data using Grafana.