-- Enabled before the app creates its tables (docker-entrypoint-initdb.d).
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
