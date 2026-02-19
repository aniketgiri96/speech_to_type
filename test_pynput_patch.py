from pynput.keyboard import GlobalHotKeys
import inspect

def test_patch(patch_func):
    original = GlobalHotKeys._on_press
    GlobalHotKeys._on_press = patch_func
    try:
        hk = GlobalHotKeys({'<ctrl>+<alt>+<space>': lambda: None})
        print(f"Success with {patch_func.__name__}")
        return True
    except Exception as e:
        print(f"Failed with {patch_func.__name__}: {e}")
        try:
            print(f"Exception type: {type(e)}")
            import threading
            # Try to reproduce the AssertionError
            str(e)
        except Exception as e2:
            print(f"Error during str(e): {e2}")
        return False
    finally:
        GlobalHotKeys._on_press = original

def patch1(self, key, injected=False, *args, **kwargs):
    pass

def patch2(self, key, injected=False):
    pass

def patch3(self, key, injected):
    pass

print("Testing patch1 (args, kwargs):")
test_patch(patch1)
print("\nTesting patch2 (named optional):")
test_patch(patch2)
print("\nTesting patch3 (named required):")
test_patch(patch3)
