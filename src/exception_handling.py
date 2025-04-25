import logging
import functools
import traceback
from typing import Callable, TypeVar, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for better typing support
T = TypeVar("T")
R = TypeVar("R")


def graceful_exceptions(
    fallback_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False,
    notify: bool = False,
    max_retries: int = 0,
) -> Callable[[Callable[..., R]], Callable[..., Optional[R]]]:
    """
    A decorator that catches exceptions in methods and handles them gracefully.

    Args:
        fallback_return: Value to return if an exception occurs
        log_level: Logging level for exceptions
        reraise: Whether to reraise the exception after handling
        notify: Whether to send notifications for critical errors
        max_retries: Number of times to retry the function before giving up

    Returns:
        Decorated function that handles exceptions gracefully
    """

    def decorator(func: Callable[..., R]) -> Callable[..., Optional[R]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[R]:
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Extract class name if method is inside a class
                    class_name = args[0].__class__.__name__ if args else ""

                    # Log the exception
                    error_message = f"Exception in {class_name}.{func.__name__}: {str(e)}"
                    if log_level == logging.DEBUG:
                        logger.debug(error_message, exc_info=True)
                    elif log_level == logging.INFO:
                        logger.info(error_message)
                    elif log_level == logging.WARNING:
                        logger.warning(error_message)
                    else:
                        logger.error(f"{error_message}\n{traceback.format_exc()}")

                    # Handle retries
                    retries += 1
                    if retries <= max_retries:
                        logger.info(f"Retrying {func.__name__} ({retries}/{max_retries})...")
                        continue

                    # Notification for critical errors
                    if notify:
                        # You could integrate with Slack, email, etc.
                        try:
                            # Example: notify_admin(error_message)
                            pass
                        except:
                            logger.error("Failed to send notification")

                    # Reraise if specified
                    if reraise:
                        raise

                    # Return fallback value
                    return fallback_return

        return wrapper

    return decorator
