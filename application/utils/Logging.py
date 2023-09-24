def Logger(name, max_MBbyte=500, log_level=None, log_file=None):
    import logging.handlers

    logger = logging.getLogger(name)
    if log_level:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    if log_file:
        # print("LOGGING TO %s" % log_file)
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_MBbyte * 1024 * 1024, backupCount=3, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

