"""The shared registry: what a config file gets told when it names something that is not there."""

import pytest

from chess_ai.registry import Registry, RegistryError


@pytest.fixture
def registry() -> Registry:
    made: Registry = Registry("widget")

    @made.register("round")
    class Round:
        def __init__(self, spec, *, radius: float = 1.0, colour: str = "red") -> None:
            self.spec, self.radius, self.colour = spec, radius, colour

    @made.register("square")
    class Square:
        def __init__(self, spec) -> None:
            self.spec = spec

    return made


def test_a_registered_name_builds_the_thing_with_its_options(registry):
    widget = registry.create("round", "a spec", radius=2.0)

    assert (widget.spec, widget.radius, widget.colour) == ("a spec", 2.0, "red")


def test_the_names_are_listed_sorted(registry):
    assert registry.names() == ["round", "square"]


def test_the_options_are_the_parameters_with_defaults(registry):
    assert registry.options("round") == ["colour", "radius"]
    assert registry.options("square") == [], "the spec it is built for is not an option"


def test_an_unknown_name_lists_what_there_is_and_guesses(registry):
    with pytest.raises(RegistryError, match="no widget called 'rond', did you mean 'round'"):
        registry.create("rond", "a spec")

    with pytest.raises(RegistryError, match=r"widgets: round, square"):
        registry.create("hexagon", "a spec")


def test_an_unknown_option_is_refused_and_guessed_at(registry):
    with pytest.raises(RegistryError, match="has no option 'raduis', did you mean 'radius'"):
        registry.create("round", "a spec", raduis=2.0)


def test_an_option_on_something_that_takes_none_says_it_takes_none(registry):
    with pytest.raises(RegistryError, match="widget 'square' has no option 'radius'$"):
        registry.create("square", "a spec", radius=2.0)


def test_what_the_constructor_needs_cannot_be_given_as_an_option(registry):
    """A config naming `spec` would be overriding what the trainer just worked out."""
    with pytest.raises(RegistryError, match="has no option 'spec'"):
        registry.create("round", "a spec", spec="another")


def test_the_name_itself_cannot_collide_with_an_option(registry):
    with pytest.raises(RegistryError, match="has no option 'name'"):
        registry.create("round", "a spec", name="round")


def test_registering_a_name_twice_is_refused(registry):
    """The name is what every config file and every checkpoint refers to."""
    with pytest.raises(RegistryError, match="widget .round. is registered twice"):

        @registry.register("round")
        class Rounder:
            pass


def test_something_taking_arbitrary_options_checks_its_own(registry):
    @registry.register("open")
    class Open:
        def __init__(self, spec, **anything) -> None:
            self.anything = anything

    assert registry.options("open") == []
    assert registry.create("open", "a spec", whatever=1).anything == {"whatever": 1}
