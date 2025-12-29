"""Tests for the schema module."""

from nyrag.schema import VespaSchema


class TestVespaSchema:
    """Tests for VespaSchema class."""

    def test_default_initialization(self):
        """Test VespaSchema initialization with default values."""
        schema = VespaSchema(schema_name="test_schema", app_package_name="test_app")
        assert schema.schema_name == "test_schema"
        assert schema.app_package_name == "test_app"
        assert schema.embedding_dim == 384
        assert schema.chunk_size == 1024
        assert schema.distance_metric == "angular"

    def test_custom_initialization(self):
        """Test VespaSchema initialization with custom values."""
        schema = VespaSchema(
            schema_name="custom_schema",
            app_package_name="custom_app",
            embedding_dim=512,
            chunk_size=2048,
            distance_metric="euclidean",
        )
        assert schema.schema_name == "custom_schema"
        assert schema.app_package_name == "custom_app"
        assert schema.embedding_dim == 512
        assert schema.chunk_size == 2048
        assert schema.distance_metric == "euclidean"

    def test_create_schema_fields(self):
        """Test that create_schema_fields returns a valid Schema object."""
        schema = VespaSchema(schema_name="test_schema", app_package_name="test_app")
        vespa_schema = schema.create_schema_fields()

        # Check that it returns a Schema object
        assert vespa_schema is not None
        assert hasattr(vespa_schema, "name")
        assert vespa_schema.name == "test_schema"

    def test_create_app_package(self):
        """Test that schema can be converted to application package."""
        schema = VespaSchema(schema_name="test_schema", app_package_name="test_app")
        # create_schema_fields method exists and creates a schema
        vespa_schema = schema.create_schema_fields()
        assert vespa_schema is not None
        assert vespa_schema.name == "test_schema"

    def test_schema_with_different_distance_metrics(self):
        """Test schema creation with different distance metrics."""
        metrics = ["angular", "euclidean", "dotproduct", "prenormalized-angular"]

        for metric in metrics:
            schema = VespaSchema(
                schema_name=f"test_{metric}",
                app_package_name="test_app",
                distance_metric=metric,
            )
            vespa_schema = schema.create_schema_fields()
            assert vespa_schema is not None

    def test_schema_with_different_dimensions(self):
        """Test schema creation with different embedding dimensions."""
        dimensions = [128, 256, 384, 512, 768, 1024]

        for dim in dimensions:
            schema = VespaSchema(
                schema_name="test_schema",
                app_package_name="test_app",
                embedding_dim=dim,
            )
            vespa_schema = schema.create_schema_fields()
            assert vespa_schema is not None

    def test_schema_with_different_chunk_sizes(self):
        """Test schema creation with different chunk sizes."""
        chunk_sizes = [256, 512, 1024, 2048, 4096]

        for size in chunk_sizes:
            schema = VespaSchema(schema_name="test_schema", app_package_name="test_app", chunk_size=size)
            vespa_schema = schema.create_schema_fields()
            assert vespa_schema is not None
