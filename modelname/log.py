# JSON string does not support for comment !

LOG_CONFIG = '''
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "file": {
            "format": "%(asctime)s: %(message)s", 
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }, 
        "screen":{
            "format": "%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "file",
            "filename": "train_report.log", 
            "mode": "w"
        },
        "screen": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "screen",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "main.py": {
            "level": "DEBUG",
            "handlers": [
                "file",
                "screen"
            ]
        },
        "graph.py": {
            "level": "DEBUG",
            "handlers": [
                "file",
                "screen"
            ]
        }
    }
 
}
'''

"""Do not add this, or things will print twice.
"root": {
        "level": "DEBUG",
        "handlers": [
            "file",
            "screen"
        ]
    }
"""
"""Old Version
"formatters": {
        "file": {
            "format": "%(asctime)s - %(name)s - %(message)s", 
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }, 
        "screen":{
            "format": "%(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
"""