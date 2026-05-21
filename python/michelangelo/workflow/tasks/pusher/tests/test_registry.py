"""Tests for the plugin registry module."""

from __future__ import annotations

from unittest import TestCase

from michelangelo.workflow.tasks.pusher.exceptions import ConfigurationError
from michelangelo.workflow.tasks.pusher.registry import PluginRegistry, default_registry


def _make_plugin_class(name: str = "MockPlugin") -> type:
    """Return a minimal mock plugin class for registry testing."""
    return type(name, (), {})


class TestPluginRegistryRegisterAndGet(TestCase):
    """Tests for PluginRegistry.register() and get()."""

    def test_register_and_get_returns_class_and_type(self):
        """It returns (plugin_class, artifact_type) after registration."""
        registry = PluginRegistry()
        cls = _make_plugin_class()
        registry.register("my_plugin", cls, list)
        result = registry.get("my_plugin")
        self.assertEqual(result, (cls, list))

    def test_register_with_none_artifact_type(self):
        """It stores None as artifact_type for config-only plugins."""
        registry = PluginRegistry()
        cls = _make_plugin_class()
        registry.register("no_type_plugin", cls)
        _, artifact_type = registry.get("no_type_plugin")
        self.assertIsNone(artifact_type)

    def test_duplicate_registration_raises_value_error(self):
        """It raises ValueError when the same name is registered twice."""
        registry = PluginRegistry()
        cls = _make_plugin_class()
        registry.register("my_plugin", cls)
        with self.assertRaises(ValueError) as ctx:
            registry.register("my_plugin", _make_plugin_class("Other"))
        self.assertIn("my_plugin", str(ctx.exception))
        self.assertIn("already registered", str(ctx.exception))

    def test_get_unknown_name_raises_configuration_error(self):
        """It raises ConfigurationError with the name listed when not found."""
        registry = PluginRegistry()
        with self.assertRaises(ConfigurationError) as ctx:
            registry.get("unknown_plugin")
        self.assertIn("unknown_plugin", str(ctx.exception))


class TestPluginRegistryParentChain(TestCase):
    """Tests for PluginRegistry parent-chain lookup via extend()."""

    def test_child_falls_through_to_parent(self):
        """It returns a parent's registration when the name is absent locally."""
        parent = PluginRegistry()
        cls = _make_plugin_class()
        parent.register("parent_plugin", cls, dict)

        child = parent.extend()
        result = child.get("parent_plugin")
        self.assertEqual(result, (cls, dict))

    def test_child_override_shadows_parent(self):
        """It returns the child's registration for a name also in the parent."""
        parent = PluginRegistry()
        parent_cls = _make_plugin_class("Parent")
        parent.register("shared_plugin", parent_cls)

        child = parent.extend()
        child_cls = _make_plugin_class("Child")
        child.register("shared_plugin", child_cls)

        child_class, _ = child.get("shared_plugin")
        self.assertIs(child_class, child_cls)

    def test_child_override_does_not_mutate_parent(self):
        """It does not change the parent's registration when the child overrides."""
        parent = PluginRegistry()
        parent_cls = _make_plugin_class("Parent")
        parent.register("shared_plugin", parent_cls)

        child = parent.extend()
        child.register("shared_plugin", _make_plugin_class("Child"))

        parent_class, _ = parent.get("shared_plugin")
        self.assertIs(parent_class, parent_cls)

    def test_get_unknown_name_raises_when_not_in_any_ancestor(self):
        """It raises ConfigurationError when the name is absent from the full chain."""
        grandparent = PluginRegistry()
        parent = grandparent.extend()
        child = parent.extend()
        with self.assertRaises(ConfigurationError):
            child.get("totally_unknown")


class TestPluginRegistryRegisteredNames(TestCase):
    """Tests for PluginRegistry.registered_names()."""

    def test_empty_registry_returns_empty_list(self):
        """It returns an empty list for a registry with no registrations."""
        self.assertEqual(PluginRegistry().registered_names(), [])

    def test_returns_sorted_local_names(self):
        """It returns sorted names registered locally."""
        registry = PluginRegistry()
        registry.register("z_plugin", _make_plugin_class())
        registry.register("a_plugin", _make_plugin_class())
        self.assertEqual(registry.registered_names(), ["a_plugin", "z_plugin"])

    def test_includes_parent_names(self):
        """It returns the union of local and parent names, sorted."""
        parent = PluginRegistry()
        parent.register("parent_plugin", _make_plugin_class())
        child = parent.extend()
        child.register("child_plugin", _make_plugin_class())
        self.assertEqual(child.registered_names(), ["child_plugin", "parent_plugin"])

    def test_deduplicates_overridden_names(self):
        """It lists an overridden name only once."""
        parent = PluginRegistry()
        parent.register("shared", _make_plugin_class())
        child = parent.extend()
        child.register("shared", _make_plugin_class("Override"))
        self.assertEqual(child.registered_names().count("shared"), 1)


class TestDefaultRegistry(TestCase):
    """Tests for the module-level default_registry."""

    def test_default_registry_is_plugin_registry_instance(self):
        """It is a PluginRegistry instance."""
        self.assertIsInstance(default_registry, PluginRegistry)

    def test_default_registry_starts_empty(self):
        """It is empty before plugins/__init__.py populates it."""
        # default_registry is populated by plugins/__init__.py (PR4).
        # This test guards that the registry object itself exists and is
        # queryable; population is tested in test_push_dispatch.py.
        self.assertIsInstance(default_registry.registered_names(), list)
