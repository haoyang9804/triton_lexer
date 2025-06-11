import os
import argparse
import subprocess
import sys

unwanted_statements = [
  "triton_dist",
  "triton.language.extra.cuda",
]

wanted_statements = [
  "@triton.jit",
]

def check_file(file_path):
    """Checks if a file contains any unwanted statements."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            for statement in unwanted_statements:
                if statement in content:
                    return True
            for statement in wanted_statements:
                if statement not in content:
                    return True
            return False
    except FileNotFoundError:
        return False

def check_python_execution(file_path):
    """检查Python文件是否能正常运行"""
    try:
        # 使用subprocess运行Python文件，捕获输出和错误
        result = subprocess.run([sys.executable, file_path], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)  # 设置10秒超时
        
        # 如果返回码不为0，说明运行出错
        if result.returncode != 0:
            print(f"Error running {file_path}:")
            print(result.stderr)
            return True  # 返回True表示需要删除
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout running {file_path}")
        return True
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return True

def process_file(file_path, dry_run=False):
    """Deletes a file if it contains unwanted statements or fails to run, unless it's a dry run."""
    should_delete = check_file(file_path) or check_python_execution(file_path)
    
    if should_delete:
        if not dry_run:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
        else:
            print(f"(Dry run) Would delete: {file_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Scans Python files for unwanted import statements and deletes them."
    )
    parser.add_argument(
        "--paths",
        nargs='+',
        help="One or more file or directory paths to scan."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without deleting any files."
    )
    args = parser.parse_args()

    for path in args.paths:
        if os.path.isfile(path):
            if path.endswith('.py'):
                process_file(path, args.dry_run)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        process_file(full_path, args.dry_run)
        else:
            print(f"Warning: '{path}' is not a valid file or directory. Skipping.")

if __name__ == '__main__':
    main()
  