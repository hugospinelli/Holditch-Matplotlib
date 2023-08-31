from typing import Any
from collections import UserList
from collections.abc import Generator

class Ring(UserList):
    """A list-like container with circular indexing."""
    
    def nwise(self, n: int, closed: bool = True) -> Generator[tuple[Any, ...]]:
        stop = len(self) if closed else len(self) - 1
        for k in range(stop):
            yield self[k: k + n]

    def pairs(self, closed: bool = True) -> Generator[tuple[Any, Any]]:
        yield from self.nwise(2, closed)

    def triples(self, closed: bool = True) -> Generator[tuple[Any, Any, Any]]:
        yield from self.nwise(3, closed)
    
    def insert(self, k: int, value: Any) -> None:
        self.data.insert(k % len(self.data), value)

    @staticmethod
    def _get_slice_range(s: slice, n: int) -> range:
        if (step := s.step) is None:
            step = 1
        elif step == 0:
            raise ValueError('slice step cannot be zero')
        if (start := s.start) is None:
            start = 0 if step > 0 else -1
        if (stop := s.stop) is None:
            stop = n if step > 0 else -n - 1
        return range(start, stop, step)

    def __getitem__(self, s: int | slice) -> Any:
        if (n := len(self)) == 0:
            raise IndexError(f'{type(self).__name__} is empty')
        if isinstance(s, slice):
            return [self.data[k % n] for k in self._get_slice_range(s, n)]
        return self.data[s % n]

    def __setitem__(self, s: int | slice, x: Any) -> None:
        if (n := len(self)) == 0:
            raise IndexError(f'{type(self).__name__} is empty')
        if isinstance(s, slice):
            for k, v in zip(self._get_slice_range(s, n), x):
                self.data[k % n] = v
        else:
            self.data[s % n] = x

    def __delitem__(self, s: int | slice) -> None:
        if (n := len(self)) == 0:
            raise IndexError(f'{type(self).__name__} is empty')
        if isinstance(s, slice):
            for k in self._get_slice_range(s, n):
                del self.data[k % n]
        del self.data[s % n]

    def __iter__(self) -> Generator[Any]:
        yield from self.data
        
    def __str__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(str, self.data))})"

    def __repr__(self) -> str:
        return f'<{self}>'
