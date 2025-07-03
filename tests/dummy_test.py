

class Dummy:
    def __init__(self, data):
        self.data = data

def setup1():
    return Dummy(2)


def manipulate1(val):
    val = manipulate2(val)
    val.data += 1
    raise RuntimeError("haha")
    return val
    
def manipulate2(val):
    val.data += 1
    return val

def test_dummy():
    val = setup1()
    val = manipulate1(val)
    val = manipulate2(val)
    assert val.data == 4


if __name__ == '__main__':
    from pathlib import Path

    from tracer import start_tracing, stop_tracing
    start_tracing(
        scope_path=Path(__file__).parent,
        main_file=str(Path(__file__)),
    )
    test_dummy()
    stop_tracing("tracing_test.json")