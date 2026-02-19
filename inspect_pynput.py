import inspect
from pynput.keyboard import GlobalHotKeys

def patch1(self, key, injected=False, *args, **kwargs):
    pass

class Fake:
    pass

f = Fake()
# Simulate a bound method
bound_method = patch1.__get__(f, Fake)

spec = inspect.getfullargspec(bound_method)
print(f"Spec for bound patch1: {spec}")
print(f"Number of args: {len(spec.args)}")

# Let's try to find pynput's _util file and read it
import pynput._util as util
print(f"pynput._util path: {util.__file__}")
