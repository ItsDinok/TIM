"""
List of arguments:
    - TEG -> Yes or No
    - Replay buffer -> Yes or No
    - TEG epochs -> number
    - Main model epochs -> number
    - TEG optimiser -> string
    - Main model optimiser -> string
    (TEG and main model optimiser come with parameters)
    - Dataset -> string
    - Tasks -> number
    - Classes per task -> number
    - Specific task constructions -> List of lists
    - Replay buffer size -> number
    - Replay buffer size -> string (class/task/total)
    - Buffer type -> string
    - metrics -> list of strings
    - file out -> string
    - transforms/preprocessing -> tbd
    - verbose -> yes/no

    - f -> input file (.txt)

These can be provided either as flags in the cmd or as a provided file.
-f must be used first if it is being used
"""

def interpret_arguments(args: str):
    # Tokenise
    args = args.split(" ")
    if args[1] == "-f":
        # TODO: Implement this
        print()