I have uploaded a python code repository in the /testbed directory.

Your task is to implement a new feature in this codebase.
First go through the codebase and identify a suitable new feature to add.
Come up with a plan to implement it and then make the necessary changes to the codebase.
You can use the tools provided to edit files and submit your changes.
Do not make edits to pre-existing tests. Pre-existing tests in the repo should be left alone.
The feature you introduce should not break any existing functionality. 
You should prioritize to re-use existing helper functions and code patterns in the repo, you can make changes to existing helper functions if needed to support the new feature.
Make sure the edit you make is complex - you should introduce at least two related changes in the codebase in different files.

Follow these steps to resolve the issue:
1. First explore the codebase to understand what it does and plan what feature maybe useful to add.
- Read documentation to understand pre-existing features as well as the structure of the repository.
- Brainstorm what feature might be useful to add and design at a high level how it would be implmented.
- Identify what part of the code would need to be modified to introduce the feature.

2. Explore the codebase to locate and understand the code relevant to the feature being added. 
- Use efficient search commands to identify key files and functions (i.e. use `grep` in the `bash` tool).  
- You should err on the side of caution and look at various relevant files and build your understanding of 
    - how the code works
    - what are the expected behaviors and edge cases

3. Implement your solution:
    - Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
    - You should be thorough and methodical.
    - The `rewrite` tool requires line numbers, if you are unsure about the line numbers, use the `view` tool to view a file with line numbers.

4. Verify your solution:
    - Make sure your code modification is syntactically correct, you can try to compile the file. 
    - Double check the code modification you have made so it correctly implements the planned feature.
    - DO NOT MODIFY any of the existing unit tests. 

5. Submit your solution:
    - Once you have verified your solution, submit your solution using the `submit` tool.

Additional recommendations:
- You should be thorough, methodical, and prioritize quality over speed. Be comprehensive.
- You should think carefully before making the tool call about what should be done. However, each step should only use one tool call. YOU SHOULD NOT USE TOOLS INSIDE YOUR THOUGHT PROCESS. 
- Each action you take is somewhat expensive, avoid repeating yourself unless necessary. 
    - Your grep commands should identify both relevant files and line numbers so you can use the file_editor tool.
    - Use grep with `-A -B -C` flags to quickly identify the relevant code blocks during your exploration.
- When exploring the codebase, use targeted search patterns to minimize unnecessary operations.
- Ensure the fix doesn't break existing functionality.
- Maximize the use of existing helper functions and code patterns in the repo, do not create new lower-level helper functions unless absolutely necessary.
