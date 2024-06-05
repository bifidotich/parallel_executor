import functools
import time


def retry(max_attempts, retry_delay_seconds, active=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    function_name = func.__name__
                    exception_description = str(e)
                    print(f"'{function_name}' raised an exception: {exception_description}")
                    attempts += 1
                    if attempts < max_attempts:
                        time.sleep(retry_delay_seconds)
                    else:
                        print(f"Reached {func.__name__} maximum number of attempts ({max_attempts}). Exiting...")
                    if not active:
                        raise
        return wrapper
    return decorator
