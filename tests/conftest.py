import warnings

# Suppress pydantic v2 deprecation warnings coming from project models
warnings.filterwarnings(
    "ignore",
    message=r".*class-based `config` is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*`@validator` validators are deprecated.*",
    category=DeprecationWarning,
)

# Suppress Starlette multipart pending deprecation
warnings.filterwarnings(
    "ignore",
    message=r"Please use `import python_multipart` instead\.",
    category=PendingDeprecationWarning,
)
