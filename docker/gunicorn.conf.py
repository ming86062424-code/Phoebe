import os

bind = "0.0.0.0:8000"
worker_class = "uvicorn_worker.UvicornWorker"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = 120
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
