from pathlib import Path
import re


def extract_test_info(entrypoint):
    """
    Extract both the file name and test function name from pytest's entrypoint.
    Handles both function-based and class-based tests.
    
    Examples:
        test_example-2.9 -> (None, test_example)
        test_separator[or] -> (None, test_separator)
        module::test_function[param] -> (module.py, test_function)
        path.to.module::test_function[param1-param2]-1.5 -> (path/to/module.py, test_function)
        tests/rules/test_indentation.py::IndentationStackTestCase::test_simple_mapping -> (tests/rules/test_indentation.py, test_simple_mapping)
    """
    # Split on "::" to separate file, class, and method
    parts = entrypoint.split("::")
    
    if len(parts) == 1:
        # No module specified, just the test function
        module_path = None
        test_name = parts[0]
    elif len(parts) == 2:
        # Module::function format
        module_part = parts[0]
        test_name = parts[1]
        
        if '/' in module_part or module_part.endswith('.py'):
            # Already a file path
            module_path = module_part
        else:
            # Dotted module path, convert to file path
            module_path = module_part.replace('.', '/') + '.py'
    else:
        # File::Class::method format (len(parts) >= 3)
        module_path = parts[0]  # First part is the file path
        test_name = parts[-1]   # Last part is the method name
        # Middle parts are class names - we ignore them for function extraction
    
    # Clean up the test function name
    # Remove pytest parameterization brackets [...]
    clean_test_name = re.sub(r'\[.*?\]', '', test_name)
    
    # Remove pytest generation suffixes like -1.2, -2.9, etc.
    clean_test_name = re.sub(r'-\d+\.\d+$', '', clean_test_name)
    
    # Remove other numeric suffixes
    clean_test_name = re.sub(r'-\d+$', '', clean_test_name)
    
    return module_path, clean_test_name


def snip_trace(events: list[dict], entrypoint: str) -> list[dict]:
    if "::" not in entrypoint:
        raise RuntimeError(f"Invalid entrypoint form: {entrypoint}")
    entry_file, entry_function = extract_test_info(entrypoint)
    found_i = None
    for i, ev in enumerate(events):
        if ev["name"] != entry_function:
            continue
        if not (entry_file is not None and ev['location'].startswith(entry_file)):
            continue

        found_i = i
        break

    if found_i is None:
        raise RuntimeError(f"Could not find entrypoint {entrypoint} in trace")
    
    return events[found_i:]


def print_function_calls(events):
    for e in events:
        if e['call_type'] == 'function_call':
            print(e['location'], e['name'])
