import re
import json

from unidiff import PatchSet

# loading in the bug data
with open('/home/msrt/data/rl_tasks/buggen_train.json') as f: 
    data = json.load(f)

with open('//home/msrt/data/rl_tasks/featadd_train.json') as f: 
    featadd_data = json.load(f)

with open('/home/msrt/data/rl_tasks/swesmith_train.json') as f:
    swesmith_data = json.load(f)

def extract_changed_files(diff_text: str):
    pattern = re.compile(r"^diff --git a/([^ ]+) b/", re.MULTILINE)
    return pattern.findall(diff_text)

def get_tests(data):
    pre_existing_tests = set()
    for item in data:
        pass_to_pass = item['PASS_TO_PASS']
        fail_to_pass = item['FAIL_TO_PASS']
        pre_existing_tests = pre_existing_tests | set(pass_to_pass)
        pre_existing_tests = pre_existing_tests | set(fail_to_pass)
    return pre_existing_tests

def remove_tests_from_diff(diff_text: str) -> str:
    """
    Remove any file diffs that touch the 'tests' folder from a git diff string.

    Args:
        diff_text (str): The full git diff text.

    Returns:
        str: The filtered diff with 'tests' folder changes removed.
    """
    # Split the diff into sections per file
    file_diff_pattern = re.compile(r"^diff --git a/.* b/.*$", re.MULTILINE)
    matches = list(file_diff_pattern.finditer(diff_text))

    filtered_diff_parts = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(diff_text)
        file_diff = diff_text[start:end]

        # Extract the file path from the diff line
        diff_line = match.group()
        # Format: diff --git a/path/to/file b/path/to/file
        file_path = diff_line.split(" b/")[1]

        if "tests" not in file_path:  # Keep only if not in tests
            filtered_diff_parts.append(file_diff)

    return "".join(filtered_diff_parts)




def old_test_fail(item, pre_existing_tests):
    fail_to_pass = set(item['FAIL_TO_PASS'])
    overlap = fail_to_pass.intersection(pre_existing_tests)
    # print(overlap)
    # if len(overlap) == 0:
    #     print("no original tests")
    if len(overlap) >= 1:
        return True
    return False

def get_num_changed_test(dataset):
    test_patches = []
    for item in dataset: 
        files_lst = extract_changed_files(item['patch'])
        for files_name in files_lst: 
            if 'tests' in files_name:
                test_patches.append(item['patch'])
                break
    return test_patches, len(test_patches)

def remove_new_tests(item, pre_existing_tests):
    fail_to_pass = set(item['FAIL_TO_PASS'])
    overlap = fail_to_pass.intersection(pre_existing_tests)
    new_item = item.copy()
    new_item['FAIL_TO_PASS'] = list(overlap)

    removed_tests_diff = remove_tests_from_diff(item['patch'])
    new_item['patch'] = removed_tests_diff
    return new_item

def remove_all_AI_generated_test(dataset):
    no_AI_dataset = []
    for item in dataset:
        files_lst = extract_changed_files(item['patch'])
        tests_changed = [file for file in files_lst if "tests" in file]

        if len(tests_changed) == 0:
            new_item = item.copy()
            no_AI_dataset.append(new_item)
    return no_AI_dataset


def clean_data(dataset, pre_existing_tests):
    clean_dataset = []
    for item in dataset:
        if old_test_fail(item, pre_existing_tests): 
            new_item = remove_new_tests(item, pre_existing_tests)
            clean_dataset.append(new_item)
    return clean_dataset

pre_existing_tests = get_tests(swesmith_data)
no_AI_d3 = remove_all_AI_generated_test(data)
no_AI_d4 = remove_all_AI_generated_test(featadd_data)
clean_d3 = clean_data(data, pre_existing_tests)
clean_d4 = clean_data(featadd_data, pre_existing_tests)

print(f"{len(no_AI_d3)=}, {len(no_AI_d4)=}, {len(clean_d3)=}, {len(clean_d4)=}")

def remove_tests_from_diff2(diff_text: str) -> str:
    return str(PatchSet(
        str(f) for f in PatchSet(diff_text)
        if not ("test" in f.path and f.is_added_file and f.path.endswith("py"))
    ))


def get_added_test_files(patch):
    return [
        f.path for f in PatchSet(patch)
        if f.is_added_file and f.path.endswith(".py") and "test" in f.path
    ]

def clean_data2(dataset):
    clean_dataset = []
    for item in dataset:
        added_tests = get_added_test_files(item['patch'])
        pre_existing_tests = [
            t for t in item["FAIL_TO_PASS"]
            if t.split("::")[0] not in added_tests
        ]
        if len(pre_existing_tests) > 0: 
            new_item = item.copy()
            new_item["FAIL_TO_PASS"] = pre_existing_tests
            new_item = remove_tests_from_diff(new_item['patch'])
            clean_dataset.append(new_item)
    return clean_dataset

clean2_d3 = clean_data2(data)
clean2_d4 = clean_data2(featadd_data)
print(f"{len(clean2_d3)=}, {len(clean2_d4)=}")

json.dump(clean2_d3, open("/home/msrt/atharv/data/clean_d3.json", "w"), indent=2)
json.dump(clean2_d4, open("/home/msrt/atharv/data/clean_d4.json", "w"), indent=2)
