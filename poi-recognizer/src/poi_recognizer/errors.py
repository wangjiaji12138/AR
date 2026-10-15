"""Errors shared by the public command boundaries."""


class POIError(Exception):
    """Carry a stable category and stage without exposing input text."""

    def __init__(self, code: str, stage: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message

    def to_dict(self) -> dict[str, str]:
        """Return the version-one error payload."""
        return {"code": self.code, "stage": self.stage, "message": self.message}
