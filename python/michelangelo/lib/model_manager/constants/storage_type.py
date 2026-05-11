"""Storage type constants."""


class StorageType:
    """Storage type constants for model storage locations.

    This class defines constants for different storage backends that can be
    used to store and retrieve model files.

    Attributes:
        LOCAL: Local filesystem storage. Model files are stored on the local
            disk accessible to the current process.
        S3: Amazon S3 (or S3-compatible) object storage. Paths follow the
            ``s3://bucket/key`` convention.
    """

    LOCAL = "local"
    S3 = "s3"
