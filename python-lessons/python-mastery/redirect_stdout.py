"""Tools to temporarily redirect stdout"""

import sys
import tempfile


class redirect_stdout:
    """Context manager class to temporarily redirect stdout

    Use this in a `with redirect_stdout(new_target) as file:` context
    """

    def __init__(self, out_file):
        self.out_file = out_file
        self.stdout = None

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self.out_file

    def __exit__(self, ty, val, tb):
        sys.stdout = self.stdout


if __name__ == "__main__":
    out_str = "Testing output"
    with tempfile.NamedTemporaryFile(mode="tr+") as out_file:
        with redirect_stdout(out_file) as file:
            print(f"Writing to {out_file.name}")
            print(out_str)
        out_file.seek(0)
        contents = out_file.read()
    print(f"Read in file contents: {contents}")
