#!/usr/bin/env bash
# Wait for MySQL to accept connections, then exec the simulation command.
#
# Schema management is the responsibility of the MySQL container's
# docker-entrypoint-initdb.d hook (mounted in docker-compose.yml), NOT this
# script — keeping the responsibility in one place avoids the two-paths-of-
# truth bug the original implementation had.

set -euo pipefail

MYSQL_HOST="${MYSQL_HOST:-mysql}"
MYSQL_PORT="${MYSQL_PORT:-3306}"
MYSQL_USER="${MYSQL_USER:-arena}"
MYSQL_DATABASE="${MYSQL_DATABASE:-ai_game_theory}"

echo "[entrypoint] waiting for mysql at ${MYSQL_HOST}:${MYSQL_PORT}…"
# Use a tiny inline Python check — PyMySQL is already in the image and reads
# the password from the environment rather than the argv, so it never appears
# on the host process listing. Connection objects are explicitly closed.
python - <<'PY'
import os, sys, time
import pymysql

host = os.environ["MYSQL_HOST"]
port = int(os.environ.get("MYSQL_PORT", "3306"))
user = os.environ["MYSQL_USER"]
password = os.environ["MYSQL_PASSWORD"]
db = os.environ["MYSQL_DATABASE"]

for attempt in range(60):
    try:
        conn = pymysql.connect(host=host, port=port, user=user,
                               password=password, database=db,
                               connect_timeout=2)
        conn.close()
        print("[entrypoint] mysql is up", flush=True)
        sys.exit(0)
    except pymysql.MySQLError:
        time.sleep(2)
print("[entrypoint] mysql did not become ready in time", file=sys.stderr)
sys.exit(1)
PY

exec "$@"
