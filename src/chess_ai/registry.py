"""Names to constructors, with the option checking that makes a config file safe to edit.

Encoders and architectures are both chosen by name in an experiment config, which is the whole
point of the registries: adding an architecture is implementing one interface and registering it
under a name, and nothing in the trainer, the evaluator or the UI has to hear about it.

The checking is here rather than in either registry because it is what a config file needs and a
Python call does not. ``create("mlp", spec, widht=512)`` is a typo somebody will make in a TOML
file, and the difference between a run that stops with "no such option ... did you mean width?"
and one that trains for two days at the default width is the difference between a usable tool
and a trap. A plain ``TypeError`` from the constructor names the argument but not the
alternatives, and nothing that reads a config file can add them afterwards.
"""

import difflib
import inspect
from collections.abc import Callable
from typing import Any


class RegistryError(Exception):
    """A name is not registered, or the options given with it are not the ones it takes."""


class Registry[T]:
    """The registered constructors for one kind of thing, such as encoders or architectures."""

    def __init__(self, what: str) -> None:
        self._what = what
        self._constructors: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Register a constructor under ``name``, as a decorator on the class itself.

        Registering the same name twice is a programming error rather than a redefinition:
        the name is what a config file and every checkpoint ever written refer to, so two
        things answering to one name is a way to load the wrong weights.
        """

        def decorate(constructor: Callable[..., T]) -> Callable[..., T]:
            if name in self._constructors:
                raise RegistryError(f"{self._what} {name!r} is registered twice")
            self._constructors[name] = constructor
            return constructor

        return decorate

    def names(self) -> list[str]:
        """Every registered name, sorted, which is what an error message and ``--help`` list."""
        return sorted(self._constructors)

    def options(self, name: str) -> list[str]:
        """The options ``name`` accepts, sorted; ``[]`` when it takes arbitrary ones."""
        return sorted(self._keywords(self._lookup(name)) or ())

    def defaults(self, name: str) -> dict[str, Any]:
        """What each of ``name``'s options is when a config does not set it.

        The companion to :meth:`options`, read from the same signature. A config file that
        documents an architecture's defaults has to get them from somewhere that cannot drift.
        """
        parameters = inspect.signature(self._lookup(name)).parameters
        return {option: parameters[option].default for option in self.options(name)}

    def create(self, name: str, /, *args: Any, **options: Any) -> T:
        """Build the thing registered as ``name``, passing ``options`` as keyword arguments.

        Everything this needs to be told is positional, so that no parameter name here can
        collide with an option a config file names. A config that says ``name = "x"`` under an
        architecture gets "has no option 'name'" rather than a ``TypeError`` about arguments it
        never heard of.
        """
        constructor = self._lookup(name)
        self._check(name, constructor, options)
        return constructor(*args, **options)

    def _lookup(self, name: str) -> Callable[..., T]:
        try:
            return self._constructors[name]
        except KeyError:
            raise RegistryError(
                f"no {self._what} called {name!r}"
                + _did_you_mean(name, self._constructors)
                + f" ({self._what}s: {', '.join(self.names())})"
            ) from None

    def _check(self, name: str, constructor: Callable[..., T], options: dict[str, Any]) -> None:
        accepted = self._keywords(constructor)
        if accepted is None:
            return  # Takes **kwargs; it checks its own options, and we cannot know them.
        for option in options:
            if option not in accepted:
                raise RegistryError(
                    f"{self._what} {name!r} has no option {option!r}"
                    + _did_you_mean(option, accepted)
                    + (f" (options: {', '.join(sorted(accepted))})" if accepted else "")
                )

    @staticmethod
    def _keywords(constructor: Callable[..., T]) -> set[str] | None:
        """``constructor``'s options, or ``None`` if it takes arbitrary keyword arguments.

        An option is a parameter with a default. A parameter without one is something the
        caller has to supply — the encoder spec an architecture is built from — and naming it
        in a config file would be overriding what the trainer just worked out, not configuring
        the model, so it is not offered and not accepted.
        """
        parameters = inspect.signature(constructor).parameters.values()
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            return None
        return {
            parameter.name
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is not inspect.Parameter.empty
        }


def _did_you_mean(given: str, known) -> str:
    """A ", did you mean 'x'?" when ``given`` is close to something in ``known``, else ""."""
    close = difflib.get_close_matches(given, known, n=1)
    return f", did you mean {close[0]!r}?" if close else ""
