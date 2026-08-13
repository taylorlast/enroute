from enroute.errors import AuthenticationError, RateLimitError, TimeoutError, is_retryable
from enroute.providers.base import classify_http_error


def test_is_retryable() -> None:
    assert is_retryable(RateLimitError("x"))
    assert is_retryable(TimeoutError("x"))
    assert not is_retryable(AuthenticationError("x"))


def test_classify_http_error() -> None:
    assert isinstance(
        classify_http_error(status_code=429, body={}, provider="openai"), RateLimitError
    )
    assert isinstance(
        classify_http_error(status_code=401, body={"error": {"message": "bad"}}, provider="openai"),
        AuthenticationError,
    )
